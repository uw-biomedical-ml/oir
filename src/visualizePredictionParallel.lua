-- plot prediction, calculate dice coef,
require 'nn'
require 'nngraph'
require 'gnuplot'
require 'lfs'
require 'model'
require 'cunn'
require 'cutorch'
require 'os'

local tnt = require 'torchnet'

cutorch.setDevice(2) -- note +1 to make it 0 indexed! sigh lua
cutorch.manualSeed(123)

local cmd = torch.CmdLine()
cmd:option('--dataDir', '/home/saxiao/oir/data/res256/', 'data directory')
cmd:option('--checkpointDir', '/home/saxiao/oir/checkpoint/res256/augment/online/', 'checkpoint directory')
cmd:option('--plotDir', '/home/saxiao/oir/plot/res256/augment/online/red/', 'plot directory')
cmd:option('--split', 'test', 'batch size')
cmd:option('--epoch', 200, 'start epoch')
cmd:option('--batchSize', 32, 'batch size')
cmd:option('--nSamples', 2, 'number of samples to draw for evaluation')
cmd:option('--nThread', 1, 'number of threads')
local opt = cmd:parse(arg)

local utils = require 'utils'

local checkpoint = torch.load(string.format("%sepoch_%s.t7", opt.checkpointDir, opt.epoch))
local net = checkpoint.model
local type = net:type()

local Loader = require 'OnlineLoader'
local loader = Loader.create(opt)
local dataSet = torch.load(string.format("%s%s.t7", opt.dataDir, opt.split))
local dataList = torch.randperm(#dataSet):long()
if opt.nSamples then
  dataList = dataList[{{1, opt.nSamples}}]
end
print(dataList)

local iter = loader:iterator(opt.split, {list = dataList})
local predictLabel, dice, outputView = nil, nil, nil
local i, j = 0, 0
for data in iter() do
  local batchSize = data.input:size(1)
  i = j + 1
  j = j + batchSize
  local input = data.input:type(type)
  local target = data.target:type(type)
  local downsampleW, downsampleH = target:size(2), target:size(3)
  local output = net:forward(input)
  local _, predict = output:max(2)
  predict = predict:squeeze()
  predict = predict:view(batchSize, -1)
  local downsampleDice = utils.diceCoef(predict, target:view(batchSize, -1), true)
  local outputBatch = output:view(batchSize, downsampleW, downsampleH, -1)
  if not predictLabel then
    predictLabel = predict.new():resize(opt.nSamples, predict:size(2))
    dice = torch.Tensor():resize(opt.nSamples)
    outputView = outputBatch.new():resize(opt.nSamples, outputBatch:size(2), outputBatch:size(3), outputBatch:size(4))
  end
  predictLabel[{{i, j}}]:copy(predict)
  dice[{{i, j}}] = downsampleDice
  outputView[{{i, j}}]:copy(outputBatch)
end

predictLabel = predictLabel:long()
outputView = outputView:float()

local function upsampleLabel(output, originalSize)
  local outputFloat = output:float()
  local nClasses = outputFloat:size(3)
  local upsampleOutput = outputFloat.new():resize(originalSize.w, originalSize.h, nClasses)
  for i = 1, nClasses do
    upsampleOutput[{{},{},i}]:copy(image.scale(outputFloat[{{},{},i}], originalSize.w, originalSize.h))
  end  
  local _, upsamplePredict = upsampleOutput:max(3)
  upsamplePredict = upsamplePredict:squeeze()
  return upsamplePredict
end


local function plotIterator()
  return tnt.ParallelDatasetIterator{
    nthread = opt.nThread,
    init = function()
      require 'torchnet'
      require 'image'
      gm = require 'graphicsmagick'
    end,
    closure = function()
      return tnt.ListDataset{
        list = dataList,
        load = function(idx)
          print(idx)
          local data = dataSet[idx]
          local input, target = data.input, data.target
          local downsampleW, downsampleH = target:size(1), target:size(2)
          local fileNameRoot = string.format("%s%s/epoch_%d_%d", opt.plotDir, opt.split, opt.epoch, idx)
          local downRawFile = string.format("%s_r.png", fileNameRoot)
          utils.drawImage(downRawFile, input)
          local downTrueFile = string.format("%s_t.png", fileNameRoot)
          utils.drawImage(downTrueFile, input, target)
          local downPredictFile = string.format("%s_p_%.3f.png", fileNameRoot, dice[idx])
          local predictLabelDraw = predictLabel[idx].new():resizeAs(predictLabel[idx]):zero()
          predictLabelDraw:maskedFill(predictLabel[idx]:eq(2), 2)
          utils.drawImage(downPredictFile, input, predictLabelDraw:view(downsampleW, -1))

          local originalRawImage = gm.Image(data.rawFilePath):toTensor('byte','RGB','DHW')
          local upsamplePredict = upsampleLabel(outputView[idx], data.originalSize)
          local originalLabelImage = gm.Image(data.labelFilePath):toTensor('byte','RGB','DHW')
          local originalLabel = utils.getLabel(originalRawImage, originalLabelImage)
          local originalLabelDraw = originalLabel.new():resizeAs(originalLabel):zero()
          originalLabelDraw:maskedFill(originalLabel:eq(2),1)
          local upsampledDc = utils.diceCoef(upsamplePredict, originalLabelDraw+1)
          local upPredictFile = string.format("%s_upsampled_%0.3f.png", fileNameRoot, upsampledDc)
          local upsamplePredictDraw = upsamplePredict.new():resizeAs(upsamplePredict):zero()
          upsamplePredictDraw:maskedFill(upsamplePredict:eq(2), 2)
          utils.drawImage(upPredictFile, originalRawImage[1], upsamplePredictDraw)

          return {idx = idx, downRawFile = downRawFile, downTrueFile = downTrueFile, downPredictFile = downPredictFile, upPredictFile = upPredictFile, originalRawFile = data.rawFilePath, originalLabelFile = data.labelFilePath, dc = dice[idx], upsampledDc = upsampledDc}
        end
      }
    end,
  }
end

local entries = {}
local plotIter = plotIterator()
local cnt = 0
for entry in plotIter() do
  cnt = cnt + 1
  print(cnt)
  table.insert(entries, entry)
end

local function compareDice(a, b)
  return a.dc < b.dc
end

-- sort by dice
table.sort(entries, compareDice)
for k, entry in pairs(entries) do
  local sortedFileRoot = string.format("%s%s/sorted/epoch_%d_", opt.plotDir, opt.split, opt.epoch)
  local sortedRaw = string.format("%s%d_r_%d.png", sortedFileRoot, k, entry.idx)
  os.rename(entry.downRawFile,sortedRaw)
  local sortedPredict = string.format("%s%d_p_%.3f.png", sortedFileRoot, k, entry.dc)
  os.rename(entry.downPredictFile, sortedPredict)
  local sortedTrue = string.format("%s%d_t.png", sortedFileRoot, k)
  os.rename(entry.downTrueFile, sortedTrue)
  local sortedUpsample = string.format("%s%d_p_upsampled_%0.3f.png", sortedFileRoot, k, entry.upsampledDc)
  os.rename(entry.upPredictFile, sortedUpsample)
  local sortedOriginalLabel = string.format("%s%d_t_original.png", sortedFileRoot, k)
  os.execute("cp \"" .. entry.originalLabelFile .. "\" \"" .. sortedOriginalLabel .. "\"")
  local sortedOriginalRaw = string.format("%s%d_r_original.png", sortedFileRoot, k)
  os.execute("cp \"" .. entry.originalRawFile .. "\" \"" .. sortedOriginalRaw .. "\"")
end
