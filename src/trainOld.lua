require 'torch'
require 'nn'
require 'optim'
require 'image'

local model = require 'model'

local cmd = torch.CmdLine()
cmd:option('--dataDir', '/data/oir/CNTF/', 'data directory')
cmd:option('--pathsFile', '/home/saxiao/oir/data/cntf.t7', 'data directory')
cmd:option('--nClasses', 2, 'number of classes')
cmd:option('--trainSize', 0.5, 'training set percentage')
cmd:option('--validateSize', 0.5, 'validate set percentage')
cmd:option('--batchSize', 2, 'batch size')
cmd:option('--patchSize', 64, 'patch size')
cmd:option('--spacing', -1, 'spacing between each patch, -1 means no overlapping, so same as the patchSize')
cmd:option('--imageW', 128, 'CNN input image width') -- 320 still works
cmd:option('--imageH', 128, 'CNN input image height')

-- training options
cmd:option('--nIterations', 10, 'patch size')
cmd:option('--learningRate', 1e-3, 'patch size')
cmd:option('--momentum', 0.9, 'patch size')
cmd:option('--saveEvery', -1, 'number of iterations every which to save the checkpoint')
cmd:option('--plotTraining', true, 'plot predictions during training')

-- gpu options
cmd:option('--gpuid', 0, 'patch size')
cmd:option('--seed', 123, 'patch size')

-- checkpoint options
cmd:option('--plotDir', '/home/saxiao/oir/plot/', 'plot directory')
cmd:option('--checkpointDir', '/home/saxiao/oir/checkpoint/', 'plot directory')

local opt = cmd:parse(arg)

-- load lib for gpu
if opt.gpuid > -1 then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  else
    print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
    print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
    print('Falling back on CPU mode')
    opt.gpuid = -1 -- overwrite user setting
  end
end

local Loader = require 'Loader'
local loader = Loader.create(opt)

local net = model.uNet(opt)

local criterion = nn.CrossEntropyCriterion()

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  net = net:cuda()
  criterion = criterion:cuda()
end

local params, grads = net:getParameters()
local trainIter = loader:iterator2("train")

local type = net:type()

local function getPredictedLabelOriginalSize(predicted, rawData)
  local rawW, rawH = rawData:size(1), rawData:size(2)
  local predictedDownsampled = torch.Tensor():resize(opt.imageW*2, opt.imageH*2, 2)
  predictedDownsampled[{{1,opt.imageW},{1,opt.imageH}}]:copy(predicted[1])
  predictedDownsampled[{{opt.imageW+1,opt.imageW*2},{1,opt.imageH}}]:copy(predicted[2])
  predictedDownsampled[{{1,opt.imageW},{opt.imageH+1,opt.imageH*2}}]:copy(predicted[3])
  predictedDownsampled[{{opt.imageW+1,opt.imageW*2},{opt.imageH+1,opt.imageH*2}}]:copy(predicted[4])
  local predictedRawsize = predictedDownsampled.new():resize(rawW, rawH, 2)
  predictedRawsize[{{},{},1}] = image.scale(predictedDownsampled[{{},{},1}], rawW, rawH)
  predictedRawsize[{{},{},2}] = image.scale(predictedDownsampled[{{},{},2}], rawW, rawH)
  local _, predictedLabel = predictedRawsize:max(3)
  predictedLabel = predictedLabel:squeeze():type(type)
  return predictedLabel
end

local function accuracyInflateToOriginalSize(predicted, rawData, rawLabel)
  local predictedLabel = getPredictedLabelOriginalSize(predicted, rawData) 
  rawLabel = rawLabel:type(type)
  local hit = torch.eq(predictedLabel, rawLabel):sum()
  return hit / rawLabel:nElement()
end

local function accuracy(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)
  local hit = torch.eq(predict, target):sum()
  return hit / target:nElement()
end

-- for two classes
local function diceCoef(output, target)
  -- 2TP/(2TP+FP+FN)
  local _, predict = output:max(2)
  local tpfp = torch.eq(predict, 1)
  tpfp = tpfp:squeeze():type(type)
  local tpfn = torch.eq(target, 1)
  tpfn = tpfn:squeeze():type(type)
  local tp = torch.eq(tpfp, tpfn)
  return 2*tp / (tpfp + tpfn)
end

local currentIter = 0
local trainAccuracy = nil
local trainAccuracyHistory = {}
local validateAccuracyHistory = {}

local function getQuadrant(output, input, b)
  output[b] = input[{{1, opt.imageW},{1, opt.imageH}}]
  output[b+1] = input[{{opt.imageW+1, opt.imageW*2}, {1, opt.imageH}}]
  output[b+2] = input[{{1, opt.imageW}, {opt.imageH+1, opt.imageH*2}}]
  output[b+3] = input[{{opt.imageW+1, opt.imageW*2}, {opt.imageH+1, opt.imageH*2}}]
end

local function downsampling(rawData, rawLabel)
  local data = torch.ByteTensor():resize(opt.batchSize*4, 1, opt.imageW, opt.imageH)
  local label = torch.ByteTensor():resize(opt.batchSize*4, opt.imageW, opt.imageH)
  local b = 1
  for i in ipairs(rawData) do
    local dataDownSampled = image.scale(rawData[i], opt.imageW*2, opt.imageH*2)
    local labelDownSampled = image.scale(rawLabel[i], opt.imageW*2, opt.imageH*2)
    getQuadrant(data[{{},1}], dataDownSampled, b)
    getQuadrant(label, labelDownSampled, b)
    b = b + 4
  end
  return data, label
end

local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()

  local rawData, rawLabel = trainIter.nextBatch()
  local data, label = downsampling(rawData, rawLabel)
  data = data:type(type)
  label = label:type(type)
  local predicted = net:forward(data)
  local labelView = label:view(label:nElement())
  trainAccuracy = accuracy(predicted, labelView)
  table.insert(trainAccuracyHistory, trainAccuracy)
  local loss = criterion:forward(predicted, labelView)
  local gradScore = criterion:backward(predicted, labelView)
  net:backward(data, gradScore)

  if opt.plotTraining then
    -- just print the first image in the batch
    local predictedView = predicted:view(opt.batchSize, opt.imageW, opt.imageH, 2)
    local predictedLabel = getPredictedLabelOriginalSize(predictedView[{{1,4}}], rawData[1], rawLabel[1])
    local rawW, rawH = rawData[1]:size(1), rawData[1]:size(2)
    local predictedImage = data.new():resize(3,rawW, rawH):zero()
    predictedImage[1]:copy(rawData[1])
    local predictedMask = predictedLabel:eq(1)
    print(predictedImage:size(), predictedMask:size())
    predictedImage[1]:maskedFill(predictedMask, 255)
    predictedImage[2]:maskedFill(predictedMask, 255)
    local fileName = opt.plotDir .. "predicted_iter" .. currentIter .. ".png"
    image.save(fileName, predictedImage)
    
    local rawFile = opt.plotDir .. "raw_iter" .. currentIter .. ".png"
    local rawImage = rawData[1].new():resize(3, rawW, rawH):zero()
    rawImage[1]:copy(rawData[1])
    image.save(rawFile, rawImage)
    
    local trueImage = data.new():resizeAs(predictedImage):zero()
    trueImage[1]:copy(rawData[1])
    local trueLabel = rawLabel[1]:type(type)
    local trueMask = trueLabel:eq(1)
    print(trueImage:type(), trueMask:type())
    trueImage[1]:maskedFill(trueMask, 255)
    trueImage[2]:maskedFill(trueMask, 255)
    local trueFileName = opt.plotDir .. "true_iter" .. currentIter .. ".png"
    image.save(trueFileName, trueImage) 
  end
  
  return loss, grads
end

local validateIter = loader:iterator2("validate")
local function validateAccuracy()
  local rawData, rawLabel = validateIter.nextBatch()
  local data, label = downsampling(rawData, rawLabel)
  data = data:type(type)
  label = label:type(type)
  local output = net:forward(data):view(opt.batchSize, opt.imageW, opt.imageH, 2)
  
  local total = 0
  local nImages = opt.batchSize / 4
  for i = 1, nImages do
    total = total + accuracyInflateToOriginalSize(output[{{4*(i-1)+1, 4*i}}], rawData[i], rawLabel[i])
  end
  return total / nImages  
end

local validateAccuracyHistory = {}
local optimOpt = {learningRate = opt.learningRate, momentum = opt.momentum}
for i = 1, opt.nIterations do
  currentIter = i
  local _, loss = optim.adagrad(feval, params, optimOpt)
  local valudateAccuracy = validateAccuracy()
  print("i=", i, " loss=", loss[1], " trainAccuracy=", trainAccuracy, " validateAccuracy=", valudateAccuracy)
  table.insert(validateAccuracyHistory, valudateAccuracy)
  if opt.saveEvery > 0 and i % opt.saveEvery == 0 then
    local checkpoint = {}
    checkpoint.iter = i
    checkpoint.model = net
    local fileName = opt.checkpointDir .. "iter_" .. i
    torch.save(fileName, checkpoint)  
  end
end

local trainHistory = {}
trainHistory.trainAccuracy = trainAccuracyHistory
trainHistory.validateAccuracy = validateAccuracyHistory
local trainHistoryFile = opt.checkpointDir .. "trainHistory.t7"
torch.save(trainHistoryFile, trainHistory)
