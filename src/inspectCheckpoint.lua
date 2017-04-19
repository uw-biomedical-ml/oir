require 'torch'
require 'nn'
require 'nngraph'
require 'gnuplot'
require 'lfs'
require 'model'
require 'cunn'
require 'cutorch'
require 'image'

cutorch.setDevice(2) -- note +1 to make it 0 indexed! sigh lua
cutorch.manualSeed(123)

local resolution = "res512"
local batchSize = 16
local rootDir = "/home/saxiao/oir/"
local Loader = require 'Loader'
local opt = {}
--opt.trainData = rootDir .. "data/" .. resolution .. "/train/"
opt.trainData = rootDir .. "data/train/"
opt.trainSize = 0.8 
local loader = Loader.create(opt)
local type = "torch.CudaTensor"
local classWeight = torch.Tensor({0.2,0.8})
local criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()

local function plotFigure(fileName, title, x, y1, y2)
  print(fileName)
--  print(x, y1)
  gnuplot.pngfigure(fileName)
  gnuplot.title(title)
  if y2 then
    gnuplot.plot({'train', x, y1, '+-'}, {'validate', x, y2, '+-'})
  else
    gnuplot.plot({'train', x, y1, '+-'})
  end
  
  gnuplot.plotflush()
end

local function plotWithErr(fileName, title, x, y1, y1err, y2, y2err)
  local file = io.open('plotData.txt', 'w')
  for i=1,x:size(1) do
    file:write(string.format('%d %f %f %f %f\n', x[i], y1[i], y1err[i], y2[i], y2err[i]))
  end
  io.close(file)

  gnuplot.pngfigure(fileName)
  gnuplot.title(title )
--  gnuplot.raw("plot 'plotData.txt' using 1:2:3: with errorbars, plot 'plotData.txt' using 1:4:5: with errorbars")
  gnuplot.raw("set yrange [0:1]")
  gnuplot.raw("plot 'plotData.txt' using 1:2:3 with errorlines title 'train', '' using 1:4:5 with errorlines title 'validate'")
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
  return files
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

local function evalDiceCoef(files, split, nSample)
  local cnt = 1
  local nfiles = #files
  local dc = torch.Tensor(nfiles * nSample)
  local loss = torch.Tensor(nfiles)
  local dataSamples, labelSamples = loader:sample(split, nSample)
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
      local nbatch = iend - istart + 1

      local output = net:forward(data)
      loss[f] = criterion:forward(output, label:view(label:nElement()))
      local _, predict = output:max(2)
      predict = predict:squeeze():type(type)
      predict = predict:view(nbatch, -1)
      label = label:view(nbatch, -1)
      for i = 1, nbatch do
        local dice = diceCoef(predict[i], label[i])
        dc[cnt] = dice
        cnt = cnt + 1
      end
    end
  end
  return dc, loss
end

local function evaluate(dataDir, plotDir, nSamples)
  local files = getFiles(dataDir)
  local dcTrain = evalDiceCoef(files, "train", nSamples)
  local dcValidate = evalDiceCoef(files, "validate", nSamples)
  local nfiles = #files  -- #files
  local epoch = torch.linspace(1, nfiles, nfiles):view(nfiles, 1)
  local xEpoch = torch.repeatTensor(epoch, 1, nSamples):view(nfiles * nSamples)
--  print(xEpoch)
  plotFigure(string.format("%sdice.png", plotDir), "dice coefficient", xEpoch, dcTrain, dcValidate)
end

local function evaluateErrorbar(dataDir, plotDir, nSamples)
  local files = getFiles(dataDir)
  local nfiles = #files
  local dcTrain, lossTrain = evalDiceCoef(files, "train", nSamples)
  local dcValidate, lossValidate = evalDiceCoef(files, "validate", nSamples)
  dcTrain = dcTrain:view(nfiles, -1)
  dcValidate = dcValidate:view(nfiles, -1)
  local dcTrainMean = dcTrain:mean(2):view(nfiles)
  local dcTrainStd = dcTrain:var(2):sqrt():view(nfiles)
  local dcValidateMean = dcValidate:mean(2):view(nfiles)
  local dcValidateStd = dcValidate:var(2):sqrt():view(nfiles)
  local epoch = torch.linspace(1, nfiles, nfiles)
  plotWithErr(string.format("%sdice_err.png", plotDir), "dice coef", epoch, dcTrainMean, dcTrainStd, dcValidateMean, dcValidateStd)
  plotFigure(string.format("%sloss.png", plotDir), "loss", epoch, lossTrain, lossValidate)
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
end

local function visualizeResult(dataDir, plotDir, split, nSample, epoch)
  local files = getFiles(dataDir)
  if not epoch then epoch = #files end
  local file = files[epoch]
  local checkpoint = torch.load(file)
  local net = checkpoint.model
  local dataSamples, labelSamples = loader:sample(split, nSample)
  local originalType = dataSamples:type()
  if dataSamples:type() ~= net:type() then
    dataSamples = dataSamples:type(net:type())
    labelSamples = labelSamples:type(net:type())
  end
  for b = 1, math.ceil(nSample/batchSize) do
    local bstart = batchSize * (b-1) + 1
    local bend = batchSize * b
    if bend > nSample then bend = nSample end
    local data = dataSamples[{{bstart, bend}}]
    local nnOutput = net:forward(data):view(bend-bstart+1, -1, 2)
    local label = labelSamples[{{bstart, bend}}]
    for i = bstart, bend do
      local fileNameRoot = string.format("%s%s/epoch_%d_%d", plotDir, split, epoch, i)
      local index = bend - i + 1
      local d, o, l = data[{index,1}]:type(originalType), nnOutput[index]:float(), label[index]:type(originalType)
      plotPrediction(d, o, l, fileNameRoot)      
    end
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
evaluateErrorbar(checkpointDir, plotDir, 100)
--visualizeResult(checkpointDir, plotDir, "train", 20, 133)
--visualizeResult(checkpointDir, plotDir .. "res512/", "validate", 10, 5)
