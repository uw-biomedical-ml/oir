require 'torch'
require 'nn'
require 'nngraph'
require 'gnuplot'
require 'lfs'
require 'model'
require 'cunn'
require 'cutorch'
require 'image'
require 'os'

cutorch.setDevice(2) -- note +1 to make it 0 indexed! sigh lua
cutorch.manualSeed(123)

local gm = require 'graphicsmagick'
local utils = require 'utils'

local nEpochs = -1
local plotDataFilePrefix = "plotData_res256_online_"
local resolution = "res256/augment/online"
local batchSize = 32
local rootDir = "/home/saxiao/oir/"
local Loader = require 'Loader'
local opt = {}
--opt.nfiles = 9290
opt.trainData = rootDir .. "data/res256/train/"
opt.testData = rootDir .. "data/res256/test/"
opt.trainSize = 0.8
--local Loader = require "OnlineLoader"
--local opt = {}
--opt.dataDir = "/home/saxiao/oir/data/res256/"
--opt.batchSize = batchSize
local loader = Loader.create(opt)
local type = "torch.CudaTensor"
local classWeight = torch.Tensor({0.2,0.8})
local criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()

local function plotFigure(fileName, title, x, y1, y2, y3)
  print(fileName)
--  print(x, y1)
  gnuplot.pngfigure(fileName)
  gnuplot.title(title)
  if y3 then
    gnuplot.plot({'train', x, y1, '+-'}, {'validate', x, y2, '+-'}, {'test', x, y3, '+-'})
  elseif y2 then
    gnuplot.plot({'train', x, y1, '+-'}, {'validate', x, y2, '+-'})
  else
    gnuplot.plot({'train', x, y1, '+-'})
  end
  
  gnuplot.plotflush()
end

local function plotWithErr(fileName, dataFileName, title, x, y1, y1err, y2, y2err, y3, y3err)
  local file = io.open(dataFileName, 'w')
  for i=1,x:size(1) do
    file:write(string.format('%d %f %f %f %f %f %f\n', x[i], y1[i], y1err[i], y2[i], y2err[i], y3[i], y3err[i]))
  end
  io.close(file)

  gnuplot.pngfigure(fileName)
  gnuplot.title(title )
--  gnuplot.raw("plot 'plotData.txt' using 1:2:3: with errorbars, plot 'plotData.txt' using 1:4:5: with errorbars")
  gnuplot.raw("set yrange [0:1]")
  gnuplot.raw("plot '" .. dataFileName .. "' using 1:2:3 with errorlines title 'train', '' using 1:4:5 with errorlines title 'validate', '' using 1:6:7 with errorlines title 'test'")
  gnuplot.raw('set key right bottom')
  gnuplot.xlabel('epoch')
  gnuplot.plotflush()

end

local function getFiles(dataDir)
  local files = {}
  for file in lfs.dir(dataDir) do
    if lfs.attributes(dataDir .. file, "mode") == "file" then
      table.insert(files,dataDir .. file)
    end
  end
  -- sort files by epoch number
  local function findNumber(str)
    local i, j = str:find("/[^/]*$")
    return tonumber(str:sub(i+7, j-3))
  end

  table.sort(files, function (a,b) return findNumber(a) < findNumber(b) end)
  local epochs = torch.Tensor():resize(#files)
  for i, file in pairs(files) do
    epochs[i] = findNumber(file)
  end

  return files, epochs
end

local function diceCoef(predict, target)
  local predictLabel = predict - 1
  local label = target - 1
  local eps = 1
  local tp = torch.cmul(predictLabel, label)
  local dice = (tp:sum()*2 + eps)/(predictLabel:sum() + label:sum() + eps)
--  print(dice)
  return dice
end

local function upsampleLabel(output, originalSize)
  local outputFloat = output:float()
  local upsampleOutput = outputFloat.new():resize(originalSize, originalSize, 2)
  upsampleOutput[{{},{},1}]:copy(image.scale(outputFloat[{{},{},1}], originalSize, originalSize))
  upsampleOutput[{{},{},2}]:copy(image.scale(outputFloat[{{},{},2}], originalSize, originalSize))
  local _, upsamplePredict = upsampleOutput:max(3)
  upsamplePredict = upsamplePredict:squeeze():type(type)
  return upsamplePredict
end

local function computeUpsampleDice(output, rawImageFile, rawLabelFile)
  local rawImage = gm.Image(rawImageFile):toTensor('byte','RGB','DHW')
  local labelImage = gm.Image(rawLabelFile):toTensor('byte','RGB','DHW')  
  local label = utils.getLabel(rawImage, labelImage):type(type)
  local upsamplePredict = upsampleLabel(output, label:size(1))
  local dc = diceCoef(upsamplePredict, label+1)
  print("upsampled dc: " .. dc)
  return dc 
end

local function testUpsample()
  local data, label, path = loader:sample("train", 1)
  local originalType = data:type()
  data = data:type(type)
  label = label:type(type)
  local sampleData, sampleLabel, samplePath = data[1], label[1], path[1]
  local checkpoint = torch.load("/home/saxiao/oir/checkpoint/res256/augment/11x/adam/epoch_40.t7")
  local net = checkpoint.model
  local output = net:forward(data)
  local _, predicted = output:max(2)
  predicted = predicted:squeeze()
  local predictedImgName = "/home/saxiao/tmp/downSampled.png"
  local d, p = sampleData[1]:type(originalType), predicted:type(originalType)
--  utils.drawImage(predictedImgName, d, p:view(d:size(1), -1)-1)
  
  print(output:size())
  local rawImage = gm.Image(samplePath.raw):toTensor('byte','RGB','DHW')
  local upsamplePredict = upsampleLabel(output:view(d:size(1), -1, 2), rawImage:size(2))
  print(upsamplePredict:min(), upsamplePredict:max())
  print(upsamplePredict:size(), upsamplePredict:type())
  local upp = upsamplePredict:type(originalType)
  local upsampleImgName = "/home/saxiao/tmp/upsampled.png"
--  utils.drawImage(upsampleImgName, rawImage[1], upp-1)
  local labelImage = gm.Image(samplePath.label):toTensor('byte','RGB','DHW')
  local label = utils.getLabel(rawImage, labelImage):type(type)
  print("dc = " .. diceCoef(upsamplePredict, label+1))
end

local function evalDiceCoefNew(files, split, nBatches)
  print(split)
  local nfiles = #files
  if nEpochs > 0 then nfiles = nEpochs end
  local dc = torch.Tensor(nfiles * nBatches * batchSize)
  local loss = torch.Tensor(nfiles * nBatches * batchSize)
  local iter = loader:iterator(split)
  local b = 0
  local samples = {}
  for sample in iter() do
    b = b + 1
    samples[b] = sample
    if b == nBatches then break end
  end
  local cnt = 1
  for f = 1, nfiles do
    local file = files[f]
    print(file)
    local checkpoint = torch.load(file)
    local net = checkpoint.model
    for b = 1, nBatches do
      local data, label = samples[b].input, samples[b].target
      data = data:type(net:type())
      label = label:type(net:type())
      local output = net:forward(data)
      local outputView = output:view(batchSize, -1, 2)
      local _, predict = output:max(2)
      predict = predict:squeeze():type(type)
      predict = predict:view(batchSize, -1)
      local labelView = label:view(batchSize, -1)
      for i = 1, batchSize do
        local dice = diceCoef(predict[i], labelView[i])
        dc[cnt] = dice
        loss[cnt] = criterion:forward(outputView[i], labelView[i])
        cnt = cnt + 1
      end
    end
  end
  return dc, loss
end

local function evalDiceCoef(files, split, nSample, computeUpsample)
  local cnt = 1
  local nfiles = #files
  if nEpochs > 0 then nfiles = nEpochs end
  local dc = torch.Tensor(nfiles * nSample)
  local loss = torch.Tensor(nfiles * nSample)
  local upsampledDc = nil
  if computeUpsample and slipt == 'test' then
    upsampledDc = torch.Tensor(nfiles * nSample)
  end
  local dataSamples, labelSamples, dataFiles = loader:sample(split, nSample)
  nSample = dataSamples:size(1)
  for f = 1, nfiles do
    local file = files[f]
    print(file)
    local checkpoint = torch.load(file) 
    local net = checkpoint.model
    if dataSamples:type() ~= net:type() then
      dataSamples = dataSamples:type(net:type())
      labelSamples = labelSamples:type(net:type())
    end
    for b = 1, math.ceil(nSample/batchSize) do
      local istart = batchSize * (b-1) + 1
      local iend = batchSize * b
      if iend > nSample then iend = nSample end
      local data = dataSamples[{{istart, iend}}]
      local label = labelSamples[{{istart, iend}}]
      local B = iend - istart + 1

      local output = net:forward(data)
      local outputView = output:view(B, -1, 2)
      local _, predict = output:max(2)
      predict = predict:squeeze():type(type)
      predict = predict:view(B, -1)
      local labelView = label:view(B, -1)
      for i = 1, B do
        local dice = diceCoef(predict[i], labelView[i])
        print("downed sampled dc: " .. dice)
        dc[cnt] = dice
        loss[cnt] = criterion:forward(outputView[i], labelView[i])
        if computeUpsample and split == 'test' then
          local dataId = istart + i - 1
          local dcUp = computeUpsampleDice(outputView[i]:view(label[i]:size(1), label[i]:size(2), 2), dataFiles[dataId].raw, dataFiles[dataId].label)
          upsampledDc[cnt] = dcUp 
        end
        cnt = cnt + 1
      end
    end
  end
  return dc, loss, upsampledDc
end

local function evaluate(checkpointDir, plotDir, nSamples)
  local files = getFiles(checkpointDir)
  local dcTrain = evalDiceCoef(files, "train", nSamples)
  local dcValidate = evalDiceCoef(files, "validate", nSamples)
  local nfiles = #files  -- #files
  local epoch = torch.linspace(1, nfiles, nfiles):view(nfiles, 1)
  local xEpoch = torch.repeatTensor(epoch, 1, nSamples):view(nfiles * nSamples)
--  print(xEpoch)
  plotFigure(string.format("%sdice.png", plotDir), "dice coefficient", xEpoch, dcTrain, dcValidate)
end

local function meanStd(input, nfiles)
  input = input:view(nfiles, -1)
  local mean = input:mean(2):view(nfiles)
  local std = input:var(2):sqrt():view(nfiles)
  return mean, std
end

local function evaluateErrorbarNew(checkpointDir, plotDir, nBatch)
  local files, epoch = getFiles(checkpointDir)
  local nfiles = #files
  if nEpochs > 0 then
    nfiles = nEpochs
    epoch = epoch[{{1, nfiles}}]
  end
  local dcTrain, lossTrain = evalDiceCoefNew(files, "train", nBatch)
  local dcValidate, lossValidate = evalDiceCoefNew(files, "validate", nBatch)
  local dcTest, lossTest = evalDiceCoefNew(files, "test", nBatch)

  local dcTrainMean, dcTrainStd = meanStd(dcTrain, nfiles)
  local dcValidateMean, dcValidateStd = meanStd(dcValidate, nfiles)
  local dcTestMean, dcTestStd = meanStd(dcTest, nfiles)
  local lossTrainMean, lossTrainStd = meanStd(lossTrain, nfiles)
  local lossValidateMean, lossValidateStd = meanStd(lossValidate, nfiles)
  local lossTestMean, lossTestStd = meanStd(lossTest, nfiles)
--  local dcTestUpsampleMean, dcTestUpsampleStd = meanStd(dcTestUpsample, nfiles)

--  local epoch = torch.linspace(1, nfiles, nfiles)
  plotWithErr(string.format("%sdice_err.png", plotDir), string.format("%sdc.txt", plotDataFilePrefix), "dice coef", epoch, dcTrainMean, dcTrainStd, dcValidateMean, dcValidateStd, dcTestMean, dcTestStd)
  plotWithErr(string.format("%sloss_err.png", plotDir), string.format("%sloss.txt", plotDataFilePrefix), "loss", epoch, lossTrainMean, lossTrainStd, lossValidateMean, lossValidateStd, lossTestMean, lossTestStd)
--  plotWithErr(string.format("%sdice_upsample_err.png", plotDir), string.format("%sdc_upsample.txt", plotDataFilePrefix), "dice coef", epoch, dcTestUpsampleMean, dcTestUpsampleStd)
end

local function evaluateErrorbar(checkpointDir, plotDir, nSamples)
  local files, epoch = getFiles(checkpointDir)
  local nfiles = #files
  if nEpochs > 0 then 
    nfiles = nEpochs 
    epoch = epoch[{{1, nfiles}}]
  end
  local dcTrain, lossTrain = evalDiceCoef(files, "train", nSamples)
  local dcValidate, lossValidate = evalDiceCoef(files, "validate", nSamples)
  local dcTest, lossTest = evalDiceCoef(files, "test", nSamples)

  local dcTrainMean, dcTrainStd = meanStd(dcTrain, nfiles)
  local dcValidateMean, dcValidateStd = meanStd(dcValidate, nfiles)
  local dcTestMean, dcTestStd = meanStd(dcTest, nfiles)
  local lossTrainMean, lossTrainStd = meanStd(lossTrain, nfiles)
  local lossValidateMean, lossValidateStd = meanStd(lossValidate, nfiles)
  local lossTestMean, lossTestStd = meanStd(lossTest, nfiles)
--  local dcTestUpsampleMean, dcTestUpsampleStd = meanStd(dcTestUpsample, nfiles)
  
--  local epoch = torch.linspace(1, nfiles, nfiles)
  plotWithErr(string.format("%sdice_err.png", plotDir), string.format("%sdc.txt", plotDataFilePrefix), "dice coef", epoch, dcTrainMean, dcTrainStd, dcValidateMean, dcValidateStd, dcTestMean, dcTestStd)
  plotWithErr(string.format("%sloss_err.png", plotDir), string.format("%sloss.txt", plotDataFilePrefix), "loss", epoch, lossTrainMean, lossTrainStd, lossValidateMean, lossValidateStd, lossTestMean, lossTestStd)
--  plotWithErr(string.format("%sdice_upsample_err.png", plotDir), string.format("%sdc_upsample.txt", plotDataFilePrefix), "dice coef", epoch, dcTestUpsampleMean, dcTestUpsampleStd)
end

local function plotPrediction(raw, nnOutput, label, fileNameRoot)
  local imageW, imageH = raw:size(1), raw:size(2)
  local _, predictFlat = nnOutput:max(2)
  local predict = predictFlat:view(imageW, imageH)
  print("predict highlighted", predict:eq(2):sum()/predict:nElement())
  local rawImage = raw.new():resize(3, imageW, imageH):zero()
  rawImage[1]:copy(raw)
  local rawFile = string.format("%s_r.png", fileNameRoot)
  image.save(rawFile, rawImage)
  
  local predictedImage = raw.new():resize(3,imageW, imageH):zero()
  predictedImage[1]:copy(raw)
  predictedImage[1]:maskedFill(predict:eq(2), 255)
  predictedImage[2]:maskedFill(predict:eq(2), 255)
  local dc = diceCoef(predict:type(label:type()), label)
  local predictedFile = string.format("%s_p_%.3f.png", fileNameRoot, dc)
  image.save(predictedFile, predictedImage)

  local trueImage = raw.new():resize(3, imageW, imageH):zero()
  trueImage[1]:copy(raw)
  trueImage[1]:maskedFill(label:eq(2), 255)
  trueImage[2]:maskedFill(label:eq(2), 255)
  local trueFile = string.format("%s_t.png", fileNameRoot) 
  image.save(trueFile, trueImage)
  return {dc = dc, rawFile = rawFile, predictedFile = predictedFile, trueFile = trueFile}
end

local function compareDice(a, b)
  return a.dc < b.dc
end

local function visualizeResult(checkpointDir, plotDir, split, nSample, epoch, sorted)
  local files = getFiles(checkpointDir)
  if not epoch then epoch = #files end
  local file = files[epoch]
  local checkpoint = torch.load(file)
  local net = checkpoint.model
  local dataSamples, labelSamples, dataFiles = loader:sample(split, nSample)
  nSample = dataSamples:size(1)
  local originalType = dataSamples:type()
  if dataSamples:type() ~= net:type() then
    dataSamples = dataSamples:type(net:type())
    labelSamples = labelSamples:type(net:type())
  end
  local samples = {}
  for b = 1, math.ceil(nSample/batchSize) do
    local bstart = batchSize * (b-1) + 1
    local bend = batchSize * b
    if bend > nSample then bend = nSample end
    local data = dataSamples[{{bstart, bend}}]
    local nnOutput = net:forward(data):view(bend-bstart+1, -1, 2)
    local label = labelSamples[{{bstart, bend}}]
    for i = bstart, bend do
      local fileNameRoot = string.format("%s%s/epoch_%d_%d", plotDir, split, epoch, i)
      local index = i - bstart + 1 -- bend - i + 1
      local d, o, l = data[{index,1}]:type(originalType), nnOutput[index]:float(), label[index]:type(originalType)
      local entry = plotPrediction(d, o, l, fileNameRoot)
      entry.idx = i
            
      local dataId = bstart + index - 1
      print("label = ", dataFiles[dataId].label)
      local rawImage = gm.Image(dataFiles[dataId].raw):toTensor('byte','RGB','DHW')
      local upsamplePredict = upsampleLabel(nnOutput[index]:view(data[index][1]:size(1), -1, 2), rawImage:size(2))
      local labelImage = gm.Image(dataFiles[dataId].label):toTensor('byte','RGB','DHW')
      local originalLabel = utils.getLabel(rawImage, labelImage):type(type)
      local upsampledDc = diceCoef(upsamplePredict, originalLabel+1)
      local upsampleImgName = string.format("%s_upsampled_%0.3f.png", fileNameRoot, upsampledDc)
      utils.drawImage(upsampleImgName, rawImage[1], upsamplePredict:type(originalType)-1)
      entry.upsampledFile = upsampleImgName
      entry.upsampledDc = upsampledDc
      entry.originalRawFile = dataFiles[dataId].raw
      entry.originalLabelFile = dataFiles[dataId].label
      print("dc = " .. entry.dc .. " upsampled dc = " .. entry.upsampledDc)
      table.insert(samples, entry)
    end
  end
  if sorted then
    local originalIndexes = torch.Tensor(nSample)
    table.sort(samples, compareDice)
    for k, entry in pairs(samples) do
      originalIndexes[k] = entry.idx
      local sortedFileRoot = string.format("%s%s/sorted/epoch_%d_", plotDir, split, epoch)
      local sortedRaw = string.format("%s%d_r_%d.png", sortedFileRoot, k, entry.idx)
      os.rename(entry.rawFile,sortedRaw)
      local sortedPredict = string.format("%s%d_p_%.3f.png", sortedFileRoot, k, entry.dc)
      os.rename(entry.predictedFile, sortedPredict)
      local sortedTrue = string.format("%s%d_t.png", sortedFileRoot, k)
      os.rename(entry.trueFile, sortedTrue)
      local sortedUpsample = string.format("%s%d_p_upsampled_%0.3f.png", sortedFileRoot, k, entry.upsampledDc)
      os.rename(entry.upsampledFile, sortedUpsample)
      local sortedOriginalLabel = string.format("%s%d_t_original.png", sortedFileRoot, k)
      os.execute("cp \"" .. entry.originalLabelFile .. "\" \"" .. sortedOriginalLabel .. "\"")
      local sortedOriginalRaw = string.format("%s%d_r_original.png", sortedFileRoot, k)
      os.execute("cp \"" .. entry.originalRawFile .. "\" \"" .. sortedOriginalRaw .. "\"")      
    end
    torch.save(string.format("originalIndex_%s.t7", split), originalIndexes)
  end
end

local function plotProgress(dataDir, plotDir)
  local files = getFiles(dataDir)
  local epoch = torch.Tensor(#files)
  local loss = torch.Tensor(#files)
  local trainAccuracy = torch.Tensor(#files)
  local validateAccuracy = torch.Tensor(#files)
  local trainDiceCoef = torch.Tensor(#files)
  local validateDiceCoef = torch.Tensor(#files)

  for i = 1, #files do
    local checkpoint = torch.load(files[i])
    epoch[i] = checkpoint.epoch
    loss[i] = checkpoint.loss
    trainAccuracy[i] = checkpoint.trainAccuracy
    validateAccuracy[i] = checkpoint.validateAccuracy
    trainDiceCoef[i] = checkpoint.trainDiceCoef
    validateDiceCoef[i] = checkpoint.validateDiceCoef
  end
 
  plotFigure(string.format("%sloss.png",plotDir), "loss", epoch, loss)
  plotFigure(string.format("%saccuracy.png", plotDir), "accuracy", epoch, trainAccuracy, validateAccuracy)
  plotFigure(string.format("%sdiceCoef.png", plotDir), "dice coef", epoch, trainDiceCoef, validateDiceCoef)
end


local checkpointDir = rootDir .. "checkpoint/" .. resolution .. "/"
local plotDir = rootDir .. "plot/" .. resolution .. "/"
--plotProgress(checkpointDir, plotDir)
--evaluateErrorbarNew(checkpointDir, plotDir, 3)
--visualizeResult(checkpointDir, plotDir, "train", 5, 60, true)
visualizeResult(checkpointDir, plotDir, "validate", -1, 150, true)
