require 'torch'
require 'nn'
require 'optim'

local model = require 'model'

local cmd = torch.CmdLine()
cmd:option('--dir', '/Users/saxiao/AI/oir/data/', 'data directory')
cmd:option('--nClasses', 2, 'number of classes')
cmd:option('--trainSize', 0.5, 'training set percentage')
cmd:option('--validateSize', 0.5, 'validate set percentage')
cmd:option('--batchSize', 8, 'batch size')
cmd:option('--patchSize', 64, 'patch size')
cmd:option('--spacing', -1, 'spacing between each patch, -1 means no overlapping, so same as the patchSize')
cmd:option('--imageW', 64, 'CNN input image width')
cmd:option('--imageH', 64, 'CNN input image height')

-- training options
cmd:option('--nIterations', 3000, 'patch size')
cmd:option('--learningRate', 1e-3, 'patch size')
cmd:option('--momentum', 0.9, 'patch size')

-- gpu options
cmd:option('--gpuid', -1, 'patch size')
cmd:option('--seed', 123, 'patch size')

-- checkpoint options
cmd:option('--plotDir', '/Users/saxiao/AI/oir/plot/', 'plot directory')

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
local trainIter = loader:iteratorDownSampled("train")

local type = net:type()

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

local trainAccuracy = nil
local trainAccuracyHistory = {}
local validateAccuracyHistory = {}

local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()

  local data, label = trainIter.nextBatch()
  data = data:type(type)
  label = label:type(type)
  local predicted = net:forward(data)
  trainAccuracy = accuracy(predicted, label)
  table.insert(trainAccuracyHistory, trainAccuracy)
  local loss = criterion:forward(predicted, label)
  local gradScore = criterion:backward(predicted, label)
  net:backward(data, gradScore)
  return loss, grads
end

local validateIter = loader:iterator("validate")
local function validateAccuracy()
  local data, label = validateIter.nextBatch()
  data = data:type(type)
  label = label:type(type)
  local output = net:forward(data)
  return accuracy(output, label)
end

local validateAccuracyHistory = {}
local optimOpt = {learningRate = opt.learningRate, momentum = opt.momentum}
for i = 1, opt.nIterations do
  local _, loss = optim.adagrad(feval, params, optimOpt)
  local valudateAccuracy = validateAccuracy()
  print("i=", i, " loss=", loss[1], " trainAccuracy=", trainAccuracy, " validateAccuracy=", valudateAccuracy)
  table.insert(validateAccuracyHistory, valudateAccuracy)
end

local trainHistory = {}
trainHistory.trainAccuracy = trainAccuracyHistory
trainHistory.validateAccuracy = validateAccuracyHistory
local trainHistoryFile = opt.plotDir .. "trainHistory.t7"
torch.save(trainHistoryFile, trainHistory)
