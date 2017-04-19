require 'torch'
require 'nn'
require 'optim'
require 'image'

local model = require 'model'

local cmd = torch.CmdLine()
cmd:option('--trainData', '/home/saxiao/oir/data/train/', 'data directory')
cmd:option('--nClasses', 2, 'number of classes')
cmd:option('--trainSize', 0.8, 'training set percentage')
cmd:option('--batchSize', 8, 'batch size')

-- training options
cmd:option('--maxEpoch', 1000, 'maxumum epochs to train')
cmd:option('--learningRate', 1e-2, 'starting learning rate')
cmd:option('--minLearningRate', 1e-7, 'minimum learning rate')
cmd:option('--momentum', 0.9, 'patch size')
cmd:option('--learningDecayRate', 0.01, 'learning rate decay rate')
cmd:option('--saveModelEvery', 1, 'save model every n epochs')

cmd:option('--saveEvery', -1, 'number of iterations every which to save the checkpoint')
cmd:option('--plotTraining', false, 'plot predictions during training')
cmd:option('--plotValidate', false, 'plot predictions during training')

-- gpu options
cmd:option('--gpuid', 0, 'patch size')
cmd:option('--seed', 123, 'patch size')

-- checkpoint options
cmd:option('--plotDir', '/home/saxiao/oir/plot/res512/', 'plot directory')
cmd:option('--checkpointDir', '/home/saxiao/oir/checkpoint/res512/', 'checkpoint directory')

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

-- TODO: use cross validation to determine?
local classWeight = torch.Tensor({0.2,0.8})
local criterion = nn.CrossEntropyCriterion()

-- ship the model to the GPU if desired
if opt.gpuid == 0 then
  net = net:cuda()
  criterion = criterion:cuda()
end

local params, grads = net:getParameters()
local type = net:type()

local currentIter = 0
local epoch = 1
local function calHits(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)
  local hit = torch.eq(predict, target):sum()
  return hit
end

local function diceCoef(predict, target)
  local predictLabel = predict - 1
  local label = target - 1
  local eps = 1
  local tp = torch.cmul(predictLabel, label)
--  print(predictLabel:type(), label:type(), tp:type())
  print(tp:sum(), predictLabel:sum(), label:sum())
  local dice = (tp:sum()*2 + eps)/(predictLabel:sum() + label:sum() + eps)
  print(dice)
  return dice 
end
-- for two classes
local function diceCoefFromNetOutput(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)  -- 1 is normal, 2 is yellow
  return diceCoef(predict, target)
  
  -- 2TP/(2TP+FP+FN)
--  local _, predict = output:max(2)
--  local tpfp = torch.eq(predict, 2)
--  tpfp = tpfp:squeeze():type(type)
--  local tpfpSum = tpfp:sum()
--  tpfp:maskedFill(tpfp:eq(0),2) -- tpfp=1, others=2
--  local tpfn = torch.eq(target, 2)  -- tpfn=1, others=0
--  tpfn = tpfn:squeeze():type(type)
--  local tpfnSum = tpfn:sum()
--  local tp = torch.eq(tpfp, tpfn)  -- tp=1, others=0
--  return tp:sum()*2 / (tpfpSum + tpfnSum)
end

local function combineQuadrants(input, i)
  local imageW, imageH = input:size(2)*2, input:size(3)*2
  local output = input.new():resize(imageW, imageH):zero()
  output[{{1, imageW/2},{1, imageH/2}}]:copy(input[i])
  output[{{1, imageW/2},{imageH/2+1, imageH}}]:copy(input[i+1])
  output[{{imageW/2+1, imageW},{1, imageH/2}}]:copy(input[i+2])
  output[{{imageW/2+1, imageW},{imageH/2+1, imageH}}]:copy(input[i+3])
  return output
end

local function plotPrediction(raw, nnOutput, label, split)
  local imageW, imageH = raw:size(2), raw:size(3)
  local _, predictFlat = nnOutput:max(2)
  local predict = predictFlat:view(-1, imageW, imageH)
  print("predict highlighted", predict:eq(2):sum()/predict:nElement())
  local iplot = math.ceil(opt.batchSize * math.random()) -- opt.batchSize
  for b = iplot, iplot do 
    local rawImage = raw.new():resize(3, imageW, imageH):zero()
    rawImage[1]:copy(raw[b])
    local rawFile = opt.plotDir .. split .. "/epoch_" .. epoch .. "_" .. b .."_r.png"
    image.save(rawFile, rawImage)
    
    local predictedImage = raw.new():resize(3,imageW, imageH):zero()
    predictedImage[1]:copy(raw[b])
    predictedImage[1]:maskedFill(predict[b]:eq(2), 255)
    predictedImage[2]:maskedFill(predict[b]:eq(2), 255)
    local dc = diceCoef(predict[b]:type(label:type()), label[b])
--    print("plot dc = ", dc)
    local predictedFile = string.format("%s%s/epoch_%d_%d_%.4f_p.png", opt.plotDir, split, epoch, b, dc)
    image.save(predictedFile, predictedImage)
    
    local trueImage = raw.new():resize(3, imageW, imageH):zero()
    trueImage[1]:copy(raw[b])
    trueImage[1]:maskedFill(label[b]:eq(2), 255)
    trueImage[2]:maskedFill(label[b]:eq(2), 255)
    local trueFile = opt.plotDir .. split ..  "/epoch_" ..  epoch .. "_" .. b .."_t.png"
    image.save(trueFile, trueImage)
  end
end

local function plotPredictionCombinedQuadrant(raw, nnOutput, label)
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
local trainHits, trainedSamples = 0, 0
local trainDiceCoefSum, trainIters = 0, 0
local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label = trainIter.nextBatch()
  local originalType = data:type()
--  print("label has yellow pixels:", label:size(), (label-1):sum()) 
  data = data:type(type)
  label = label:type(type) 
  local output = net:forward(data)
  local labelView = label:view(label:nElement())
  local hits = calHits(output, labelView)
  trainHits = trainHits + hits 
  trainedSamples = trainedSamples + label:nElement()
  trainDiceCoefSum = trainDiceCoefSum + diceCoefFromNetOutput(output, labelView)
  trainIters = trainIters + 1
  local loss = criterion:forward(output, labelView)
  
  local dloss = criterion:backward(output, labelView)
  net:backward(data, dloss)
  
  if opt.plotTraining and trainIter.epoch == epoch then
    print("true highlighted = ", label:eq(2):sum()/label:nElement())
    local d, o, l = data[{{},1}]:type(originalType), output:float(), label:type(originalType)
    plotPrediction(d, o, l, "train")
  end
  
  return loss, grads
end

local function validate()
  local validateIter = loader:iterator("validate")
  local hits, n = 0, 0
  local diceCoefSum, iters = 0, 0
  while validateIter.epoch < 1 do
    local data, label = validateIter.nextBatch()
    local originalType = data:type()
    data = data:type(type)
    label = label:type(type)
    local output = net:forward(data)
    local labelView = label:view(label:nElement())
    hits = hits + calHits(output, labelView)
    n = n + label:nElement()
    diceCoefSum = diceCoefSum + diceCoefFromNetOutput(output, labelView)
    iters = iters + 1
    if iters == 1 and opt.plotValidate then
      local d, o, l = data[{{},1}]:type(originalType), output:float(), label:type(originalType)
      print("validate!")
      plotPrediction(d, o, l, "validate")
    end
  end
  return hits/n, diceCoefSum/iters
end

local lr = opt.learningRate
local optimOpt = {learningRate = lr, momentum = opt.momentum}
while epoch <= opt.maxEpoch do
  currentIter = currentIter + 1
  local _, loss = optim.adagrad(feval, params, optimOpt)
  print("iter=", currentIter, " loss=", loss[1])
  
  if trainIter.epoch == epoch then
    local trainAccuracy = trainHits / trainedSamples
    local trainDiceCoef = trainDiceCoefSum / trainIters
    local validateAccuracy, validateDiceCoef = validate()
    print("epoch=",epoch," loss=", loss[1], " trainAccuracy=", trainAccuracy, " validateAccuracy=", validateAccuracy, "trainDC=", trainDiceCoef, "validateDC=", validateDiceCoef)
    local checkpoint = {}
    checkpoint.epoch = epoch
    checkpoint.iter = currentIter
    checkpoint.loss = loss[1]
    checkpoint.trainAccuracy = trainAccuracy
    checkpoint.validateAccuracy = validateAccuracy
    checkpoint.trainDiceCoef = trainDiceCoef
    checkpoint.validateDiceCoef = validateDiceCoef
    if (epoch+1) % opt.saveModelEvery == 0 then
      net:clearState()
      checkpoint.model = net
    end
    local fileName = string.format("%sepoch_%d.t7",opt.checkpointDir, epoch)
    torch.save(fileName, checkpoint)

    trainHits = 0
    trainedSamples = 0
    epoch = epoch + 1
    
    collectgarbage()
  end
  if lr > opt.minLearningRate and opt.learningDecayRate and opt.learningDecayRate > 0 then
    lr = lr * (1 - opt.learningDecayRate)
  end
end
