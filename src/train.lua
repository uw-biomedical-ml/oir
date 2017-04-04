require 'torch'
require 'nn'
require 'optim'
require 'image'

local model = require 'model'

local cmd = torch.CmdLine()
cmd:option('--trainData', '/home/saxiao/eclipse/workspace/oir/data/cntf/', 'data directory')
cmd:option('--nClasses', 2, 'number of classes')
cmd:option('--trainSize', 0.7, 'training set percentage')
cmd:option('--batchSize', 1, 'batch size')

-- training options
cmd:option('--nIterations', 1000, 'patch size')
cmd:option('--learningRate', 1e-3, 'patch size')
cmd:option('--minLearningRate', 1e-5, 'minimum learning rate')
cmd:option('--momentum', 0.9, 'patch size')
cmd:option('--learningDecayRate', 0.01, 'learning rate decay rate')

cmd:option('--saveEvery', -1, 'number of iterations every which to save the checkpoint')
cmd:option('--plotTraining', true, 'plot predictions during training')

-- gpu options
cmd:option('--gpuid', 0, 'patch size')
cmd:option('--seed', 123, 'patch size')

-- checkpoint options
cmd:option('--plotDir', '/home/saxiao/eclipse/workspace/oir/plot/', 'plot directory')
cmd:option('--checkpointDir', '/home/saxiao/eclipse/workspace/oir/checkpoint/', 'checkpoint directory')

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
local type = net:type()

local currentIter = 0
local function calAccuracy(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)
  local hit = torch.eq(predict, target):sum()
  return hit / target:nElement()
end

local function combineQuadrants(input, i)
  local imageW, imageH = input:size(2)*2, input:size(3)*2
  local output = input.new():resize(imageW, imageH):zero()
  output[{{1, imageW/2},{1, imageH/2}}]:copy(input[i])
  output[{{imageW/2+1, imageW},{1, imageH/2}}]:copy(input[i+1])
  output[{{1, imageW/2},{imageH/2+1, imageH}}]:copy(input[i+2])
  output[{{imageW/2+1, imageW},{imageH/2+1, imageH}}]:copy(input[i+3])
  return output
end

local function plotPrediction(raw, nnOutput, label)
  local imageW, imageH = raw:size(2)*2, raw:size(3)*2
  local _, predict = nnOutput:max(2)
  print("predict highlighted", predict:eq(2):sum()/predict:nElement())
  local predictedLabel = predict:eq(2):view(opt.batchSize*4, imageW/2, imageH/2)  -- 1 is hightlighted (abnormal), 0 is normal
  local i = 1
  for b = 1, opt.batchSize do
    local combinedRaw = combineQuadrants(raw, i)
    local combinedPredictedLabel = combineQuadrants(predictedLabel, i)
    local predictedImage = raw.new():resize(3, imageW, imageH):zero()
    predictedImage[1]:copy(combinedRaw)
    predictedImage[1]:maskedFill(combinedPredictedLabel, 255)
    predictedImage[2]:maskedFill(combinedPredictedLabel, 255)
    
    local combinedLabel = combineQuadrants(label, i)
    local trueImage = raw.new():resize(3, imageW, imageH):zero()
    trueImage[1]:copy(combinedRaw)
    trueImage[1]:maskedFill(combinedLabel:eq(2), 255)
    trueImage[2]:maskedFill(combinedLabel:eq(2), 255)
    i = i + 4
    
    local rawImage = raw.new():resize(3, imageW, imageH):zero()
    rawImage[1]:copy(combinedRaw)
    local rawFile = opt.plotDir .. "iter_" .. currentIter .. "_" .. b .."_r.png"
    image.save(rawFile, rawImage)
    local predictedFile = opt.plotDir .. "iter_" .. currentIter .. "_" .. b .."_p.png"
    image.save(predictedFile, predictedImage)
    local trueFile = opt.plotDir .. "iter_" .. currentIter .. "_" .. b .."_t.png"
    image.save(trueFile, trueImage)
  end
end

local trainIter = loader:iterator("train")
local trainAccuracy = nil
local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label = trainIter.nextBatch()
  local originalType = data:type()
  
  data = data:cuda()
  label = label:cuda()
  
  print("true highlighted = ", label:eq(2):sum()/label:nElement())
--    local d = data[{{},1}]:type(originalType)
--  dataimage = d.new():resize(3,data:size(3), data:size(4)):zero()
--  dataimage[1] = d[1]
--  image.save("/home/saxiao/tmp/tmp.png", dataimage)
  
  local output = net:forward(data)
  local labelView = label:view(label:nElement())
  trainAccuracy = calAccuracy(output, labelView)
  local loss = criterion:forward(output, labelView)
  
  local dloss = criterion:backward(output, labelView)
  net:backward(data, dloss)
  
  if opt.plotTraining then
    local d, o, l = data[{{},1}]:type(originalType), output:float(), label:type(originalType)
    plotPrediction(d, o, l)
  end
  
  return loss, grads
end

local validateIter = loader:iterator("validate")
local function validate()
  local data, label = validateIter.nextBatch()
  data = data:type(type)
  label = label:type(type)
  local output = net:forward(data)
  return calAccuracy(data, label:view(label:nElement()))
end

local lr = opt.learningRate
local optimOpt = {learningRate = lr, momentum = opt.momentum}
for i = 1, opt.nIterations do
  currentIter = i
  local _, loss = optim.adagrad(feval, params, optimOpt)
  print("i=", i, " loss=", loss[1], " trainAccuracy=", trainAccuracy, " validateAccuracy=", validate())
  
  if lr > opt.minLearningRate and opt.learningDecayRate and opt.learningDecayRate > 0 then
    lr = lr * (1 - opt.learningDecayRate)
  end
end
