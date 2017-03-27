require 'torch'
require 'nn'
require 'optim'

local cmd = torch.CmdLine()
cmd:option('--dir', '/Users/saxiao/AI/oir/data/', 'data directory')
cmd:option('--trainSize', 0.5, 'training set percentage')
cmd:option('--validateSize', 0.5, 'validate set percentage')
cmd:option('--batchSize', 10, 'batch size')
cmd:option('--patchSize', 64, 'patch size')
cmd:option('--spacing', -1, 'spacing between each patch, -1 means no overlapping, so same as the patchSize')

-- training options
cmd:option('--nIterations', 100, 'patch size')
cmd:option('--learningRate', 1e-3, 'patch size')
cmd:option('--momentum', 0.9, 'patch size')

local nClasses = 2

local opt = cmd:parse(arg)

local Loader = require 'Loader'
local loader = Loader.create(opt)

local net = nn.Sequential()
-- contracting path
net:add(nn.SpatialConvolutionMM(1,64,3,3,1,1,1,1)) -- 64 filters, (patchSize-3+1*2)/1+1 = patchSize, output 64 x patchSize x patchSize
net:add(nn.ReLU())
net:add(nn.SpatialConvolutionMM(64,64,3,3,1,1,1,1)) -- 64 filters, output 64 x patchSize x patchSize
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2)) -- output: 64 x patchSize/2 x patchSize/2
net:add(nn.SpatialConvolutionMM(64,128,3,3,1,1,1,1)) -- 128 filters, output 128 x patchSize/2 x patchSize/2
net:add(nn.ReLU())
net:add(nn.SpatialConvolutionMM(128,128,3,3,1,1,1,1)) -- 128 filters, output 128 x patchSize/2 x patchSize/2
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2)) -- output: 128 x patchSize/4 x patchSize/4
net:add(nn.SpatialConvolutionMM(128,256,3,3,1,1,1,1)) -- 256 filters, output 256 x patchSize/4 x patchSize/4
net:add(nn.ReLU())
net:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1)) -- 256 filters
net:add(nn.ReLU())
-- expansive path
net:add(nn.SpatialUpSamplingNearest(2)) -- output: 256 x patchSize/2 x patchSize/2
net:add(nn.SpatialConvolutionMM(256,128,3,3,1,1,1,1)) -- 128 filters, output 128 x patchSize/2 x patchSize/2
net:add(nn.ReLU())
net:add(nn.SpatialConvolutionMM(128,128,3,3,1,1,1,1))
net:add(nn.ReLU())
net:add(nn.SpatialUpSamplingNearest(2)) -- output: 128 x patchSize x patchSize
net:add(nn.SpatialConvolutionMM(128,64,3,3,1,1,1,1)) -- 64 filters, output 64 x patchSize x patchSize
net:add(nn.ReLU())
net:add(nn.SpatialConvolutionMM(64,64,3,3,1,1,1,1)) -- 64 filters 
net:add(nn.ReLU())
-- make the right shape for class prediction
net:add(nn.SpatialConvolutionMM(64,nClasses,1,1,1,1,0,0)) -- (patchSize-1)/1+1 = patchSize, output nClasses x patchSize x patchSize
net:add(nn.Reshape(nClasses, -1)) -- output batchsize x nClasses x (number of pixels)
net:add(nn.Transpose({2,3})) -- output batchsize x (number of pixels) x nClasses
net:add(nn.Reshape(-1, nClasses, false)) -- output (batchsize* number of pixels) x nClasses

local criterion = nn.CrossEntropyCriterion()

local params, grads = net:getParameters()
local trainIter = loader:iterator("train")

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

local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()

  local data, label = trainIter.nextBatch()
  local predicted = net:forward(data)
  trainAccuracy = accuracy(predicted, label)
  local loss = criterion:forward(predicted, label)
  local gradScore = criterion:backward(predicted, label)
  net:backward(data, gradScore)

  return loss, grads
end

local validateIter = loader:iterator("validate")
local function validateAccuracy()
  local data, label = validateIter.nextBatch()
  local output = net:forward(data)
  return accuracy(output, label)
end

local optimOpt = {learningRate = opt.learningRate, momentum = opt.momentum}
for i = 1, opt.nIterations do
  local _, loss = optim.adagrad(feval, params, optimOpt)
  print("loss=", loss[1], " trainAccuracy=", trainAccuracy, " validateAccuracy=", validateAccuracy())
end

