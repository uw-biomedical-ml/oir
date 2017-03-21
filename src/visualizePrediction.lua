-- plot prediction, calculate dice coef,
require 'nn'
require 'nngraph'
require 'gnuplot'
require 'lfs'
require 'cunn'
require 'cutorch'
require 'os'

local gm = require 'graphicsmagick'

cutorch.setDevice(1) -- note +1 to make it 0 indexed! sigh lua
cutorch.manualSeed(123)

local eps = 1
local cmd = torch.CmdLine()
cmd:option('--dataDir', '/home/saxiao/oir/data/res256/', 'data directory')
cmd:option('--split', 'test', 'split')
cmd:option('--checkpointDir', '/home/saxiao/oir/checkpoint/res256/augment/online/yellow/', 'checkpoint directory')  -- red:"/home/saxiao/oir/checkpoint/red/res512/", yellow: "/home/saxiao/oir/checkpoint/res256/augment/online/yellow/", red(256): '/home/saxiao/oir/checkpoint/res256/augment/online/red/'
cmd:option('--plotDir', '/home/saxiao/oir/plot/yellow/oldrun/', 'plot directory')
cmd:option('--epoch', 150, 'start epoch') -- red(512):700, red(256): 256, yellow: 150
cmd:option('--iter', 17160, 'start iter')
cmd:option('--batchSize', 32, 'batch size')
cmd:option('--nSamples', -1, 'number of samples to draw for evaluation')
cmd:option('--targetLabel', 1, 'label for the target class, yellow = 1, red = 2')
local opt = cmd:parse(arg)
local sorted = true
local plotOriginal = true
local utils = require 'src/utils'

paths.mkdir(string.format("%s/%s", opt.plotDir, opt.split))
if sorted then
  paths.mkdir(string.format("%s/%s/sorted", opt.plotDir, opt.split))
end
local checkpoint = torch.load(string.format("%sepoch_%s.t7", opt.checkpointDir, opt.epoch))
--local checkpoint = torch.load(string.format("%sepoch_%s_iter_%d.t7", opt.checkpointDir, opt.epoch, opt.iter))
local net = checkpoint.model
local type = net:type()
net:evaluate()

local Loader = require 'src/OnlineLoader'
local loader = Loader.create(opt)

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

local cnt = 0
local entries = {}
local dataList = nil
--local dataList = torch.range(21,22):long()
if opt.nSamples > 0 then
  dataList = torch.range(1, opt.nSamples):long()
end
local iterOpt = {ordered = true, list = dataList, classId = opt.targetLabel}
if split == 'control' then iterOpt.shuffleOff = true end
local iter = loader:iterator(opt.split, iterOpt)
for data in iter() do
  local dataOriginalType = data.input:type()
  local input = data.input:type(type)
  local target = data.target:type(type)
  local batchSize = input:size(1)
  local downsampleW, downsampleH = target:size(2), target:size(3)
  local output = net:forward(input)
  local _, predict = output:max(2)
  predict = predict:squeeze()
  predict = predict:view(batchSize, -1)
  local dice = utils.diceCoef(predict, target:view(batchSize, -1), 1, true)
  local outputView = output:view(batchSize, downsampleW, downsampleH, -1)
  if torch.type(dice) == 'number' then dice = torch.Tensor{dice} end
  for i = 1, dice:size(1) do
    cnt = cnt + 1
    print(cnt, data.idx[i])
    local fileNameRoot = string.format("%s%s/epoch_%d_%d", opt.plotDir, opt.split, opt.epoch, cnt)
    local downRawFile = string.format("%s_r_%d.png", fileNameRoot,data.idx[i])
    utils.drawImage(downRawFile, input[i][1]:type(dataOriginalType))
    --local dRetinaArea = utils.getRetinaArea(input[i][1])
    --local dTrueArea = 0
    local downTrueFile = nil
    if opt.split ~= 'control' then 
      local trueDraw = target.new():resizeAs(target[i]):zero()
      trueDraw:maskedFill(target[i]:eq(2),opt.targetLabel)
      --dTrueArea = trueDraw:eq(opt.targetLabel):sum()
      downTrueFile = string.format("%s_t.png", fileNameRoot)
      utils.drawImage(downTrueFile, input[i][1]:type(dataOriginalType), trueDraw:type(dataOriginalType))
    end
    local predictDraw = predict[i].new():resizeAs(predict[i]):zero()
    predictDraw:maskedFill(predict[i]:eq(2), opt.targetLabel)
    local dPredictArea = predictDraw:eq(opt.targetLabel):sum()
    local downPredictFile = string.format("%s_p_%.3f.png", fileNameRoot, dice[i])
    utils.drawImage(downPredictFile, input[i][1]:type(dataOriginalType), predictDraw:view(downsampleW, -1):type(dataOriginalType))
        
    
    local originalRawImage = gm.Image(data.rawFilePath[i]):toTensor('byte','RGB','DHW')
    local upsamplePredict = upsampleLabel(outputView[i], data.originalSize[i])
    local originalLabelDraw = nil 
    if opt.split ~= 'control' then
      local originalLabelImage = gm.Image(data.labelFilePath[i]):toTensor('byte','RGB','DHW')
      local originalLabel = utils.getLabel(originalRawImage, originalLabelImage):type(type)
      originalLabelDraw = originalLabel.new():resizeAs(originalLabel):zero()
      originalLabelDraw:maskedFill(originalLabel:eq(opt.targetLabel),1)
    else 
      originalLabelDraw = originalRawImage.new():resizeAs(originalRawImage[1]):zero()
    end
    local upsampledDc = utils.diceCoef(upsamplePredict, originalLabelDraw+1, 1)
    local upsamplePredictDraw = upsamplePredict.new():resizeAs(upsamplePredict):zero()
    upsamplePredictDraw:maskedFill(upsamplePredict:eq(2), opt.targetLabel)
    --local upRetinaArea = utils.getRetinaArea(originalRawImage[1])
    --local upTrueArea = originalLabelDraw:sum()
    --local upPredictArea = upsamplePredictDraw:eq(opt.targetLabel):sum()
    local upPredictFile = string.format("%s_upsampled_%0.3f.png", fileNameRoot, upsampledDc)
    utils.drawImage(upPredictFile, originalRawImage[1], upsamplePredictDraw)
    
    entries[cnt] = {idx = data.idx[i], downRawFile = downRawFile, downTrueFile = downTrueFile, downPredictFile = downPredictFile, upPredictFile = upPredictFile, originalRawFile = data.rawFilePath[i], originalLabelFile = data.labelFilePath[i], dc = dice[i], upsampledDc = upsampledDc}
  end
end

local function compareDice(a, b)
  return a.dc < b.dc
end

-- sort by dice
if sorted then
table.sort(entries, compareDice)
for k, entry in pairs(entries) do
  local sortedFileRoot = string.format("%s%s/sorted/epoch_%d_", opt.plotDir, opt.split, opt.epoch)
  local sortedRaw = string.format("%s%d_r_%d.png", sortedFileRoot, k, entry.idx)
  os.execute("cp \"" .. entry.downRawFile .. "\" \"" .. sortedRaw .. "\"")
  --os.rename(entry.downRawFile,sortedRaw)
  local sortedPredict = string.format("%s%d_p_%.3f.png", sortedFileRoot, k, entry.dc)
  os.execute("cp \"" .. entry.downPredictFile .. "\" \"" .. sortedPredict .. "\"")
  --os.rename(entry.downPredictFile, sortedPredict)
  local sortedUpsample = string.format("%s%d_p_upsampled_%0.3f.png", sortedFileRoot, k, entry.upsampledDc)
  os.execute("cp \"" .. entry.upPredictFile .. "\" \"" .. sortedUpsample .. "\"")
  --os.rename(entry.upPredictFile, sortedUpsample)
  if opt.split ~= 'control' then
    local sortedTrue = string.format("%s%d_t.png", sortedFileRoot, k)
    os.execute("cp \"" .. entry.downTrueFile .. "\" \"" .. sortedTrue .. "\"")
    --os.rename(entry.downTrueFile, sortedTrue)
    if plotOriginal then
      local sortedOriginalLabel = string.format("%s%d_t_original.png", sortedFileRoot, k)
      os.execute("cp \"" .. entry.originalLabelFile .. "\" \"" .. sortedOriginalLabel .. "\"")
    end
  end
  if plotOriginal then
    local sortedOriginalRaw = string.format("%s%d_r_original.png", sortedFileRoot, k)
    os.execute("cp \"" .. entry.originalRawFile .. "\" \"" .. sortedOriginalRaw .. "\"")
  end
end
end
