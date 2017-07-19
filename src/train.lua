-- rewrite with torchnet OptimEngine
require 'torch'
require 'nn'
require 'optim'
require 'image'

local model = require 'model'
local resolution = "res512"
local rootDir = "/home/saxiao/oir/"
local modelId = "res512"
local checkpointEpoch, checkpointIter = 530, 45580  -- 221300
local cmd = torch.CmdLine()
cmd:option('--trainData', rootDir .. "data/" .. resolution .. "/train/", 'data directory')
cmd:option('--testData', rootDir .. "data/res256/test/", 'test data directory')
cmd:option('--nClasses', 2, 'number of classes')
cmd:option('--trainSize', 0.8, 'training set percentage')
cmd:option('--batchSize', 8, 'batch size')
cmd:option('--targetLabel', 2, 'target label, yellow is 1, red is 2')
cmd:option('--useLocation', false, 'add location in the input')

cmd:option('--model', '/home/saxiao/oir/checkpoint/red/' .. modelId .. '/epoch_' .. checkpointEpoch .. '_iter_' .. checkpointIter .. '.t7', 'a checkpoint file')
cmd:option('--trainPatch', false, 'train by randomly selecting a patch from the original image')
cmd:option('--includeControl', false, 'including the control images')
cmd:option('--patchSize', 256, 'size of the patch')
cmd:option('--dataDir', string.format("%sdata/%s/", rootDir, resolution), 'data directory')
cmd:option('--highRes', '/home/saxiao/oir/data/res2048/', 'high resolution label directory')
cmd:option('--fullSizeDataDir', '/home/saxiao/oir/data/fullres', 'full size data directory')
-- training options
cmd:option('--maxEpoch', 3000, 'maxumum epochs to train')
cmd:option('--learningRate', 1e-2, 'starting learning rate')
cmd:option('--minLearningRate', 1e-7, 'minimum learning rate')
cmd:option('--momentum', 0.9, 'patch size')
cmd:option('--learningDecayRate', 0.01, 'learning rate decay rate')

cmd:option('--saveModelEvery', 10, 'save model every n epochs')
cmd:option('--historyFilePrefix', '/home/saxiao/oir/' .. modelId, 'prefix of the file to save the loss and accuracy for each iteration while training')
cmd:option('--validateEvery', 500, 'run validation every n iterations')
cmd:option('--trainAverageEvery', 50, 'average training metric every n iterations') 
cmd:option('--saveEvery', -1, 'number of iterations every which to save the checkpoint')
cmd:option('--plotTraining', false, 'plot predictions during training')
cmd:option('--plotValidate', false, 'plot predictions during training')

-- gpu options
cmd:option('--gpuid', 0, 'patch size')
cmd:option('--seed', 123, 'patch size')

-- checkpoint options
cmd:option('--plotDir', rootDir .. "plot/" .. resolution .. "/augment/online/", 'plot directory')
--cmd:option('--checkpointDir', rootDir .. "checkpoint/" .. resolution .. "/augment/online/yellow/control_0.5/", 'checkpoint directory')
cmd:option('--checkpointDir', rootDir .. "checkpoint/red/" .. modelId .. "/", 'checkpoint directory')

local opt = cmd:parse(arg)

local nFiles = {train=682, validate=171, test=214}

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

local Loader = require 'OnlineLoader'
local loader = Loader.create(opt)

local net = nil
if opt.model then
  local cp = torch.load(opt.model)
  net = cp.model
  optimOpt = cp.optimOpt
else
  if opt.useLocation then
    net = model.uNet1WithLocation(opt)
  else
    net = model.uNet1For512(opt)
  end
  optimOpt = {learningRate = opt.learningRate}
end

-- TODO: use cross validation to determine?
local classWeight = torch.Tensor({0.2,0.8})
local criterion = nn.CrossEntropyCriterion()

-- ship the model to the GPU if desired
if opt.gpuid > -1 then
  net = net:cuda()
  criterion = criterion:cuda()
end

local params, grads = net:getParameters()
local type = net:type()

local currentIter = 0 + checkpointIter
--local epoch = 1
local function calHits(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)
  local hit = torch.eq(predict, target):sum()
  return hit
end

-- N classes, not including background
local function diceCoef(predict, label)
--  local predictLabel = predict - 1
--  local label = target - 1
  local n = label:max()
  local a, b, c = 0, 0, 0
  local eachDice = torch.Tensor(n-1)
  local eps = 1
  for i = 2, n do
    local pi = predict:eq(i)
    local ti = label:eq(i)
    local pt = torch.cmul(pi, ti):sum()
    local psum = pi:sum()
    local tsum = ti:sum()
    eachDice[i-1] = (2*pt + eps)/(psum + tsum + eps)
    --print(string.format("class %d: %0.3f, dc = %0.3f", i, tsum/label:nElement(), eachDice[i-1]))
    a = a + pt
    b = b + psum
    c = c + tsum
  end
  local dice = (2*a + eps)/(b + c + eps)
  --print(dice)
  return dice, eachDice
end

local function diceCoef2Classes(predict, target)
  local predictLabel = predict - 1
  local label = target - 1
  local eps = 1
  local tp = torch.cmul(predictLabel, label)
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
  --print("predict highlighted", predict:eq(2):sum()/predict:nElement())
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

local sample = nil
--local trainHits, trainedSamples = 0, 0
--local trainDiceCoefSum, trainIters = 0, 0
--local trainEachDiceSum = nil
local trainLoss = {}
local trainDC = {}
local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label, location = sample.input, sample.target, sample.location
  local originalType = data:type()
--  print("label has yellow pixels:", label:size(), (label-1):sum()) 
  data = data:type(type)
  label = label:type(type)
  local output = nil
  if opt.useLocation then 
    location = location:type(type)
    output = net:forward({data, location})
  else
    output = net:forward(data)
  end
  local labelView = label:view(label:nElement())
  local hits = calHits(output, labelView)
  local dice, eachDice = diceCoefFromNetOutput(output, labelView)
  local loss = criterion:forward(output, labelView)
  table.insert(trainLoss, loss)
  table.insert(trainDC, dice)
  
  local dloss = criterion:backward(output, labelView)
  if opt.useLocation then
    net:backward({data, location}, dloss)
  else
    net:backward(data, dloss)
  end
  
  --if opt.plotTraining and trainIter.epoch == epoch then
  --  print("true highlighted = ", label:eq(2):sum()/label:nElement())
  --  local d, o, l = data[{{},1}]:type(originalType), output:float(), label:type(originalType)
  --  plotPrediction(d, o, l, "train")
  --end
 
  --local trainHistoryFile = io.open(string.format("%s_train.txt", opt.historyFilePrefix), 'a')
  --local toLog = string.format("%d %.3f %.3f\n", currentIter, loss, dice)
  --print(toLog)
  --trainHistoryFile:write(toLog)
  --io.close(trainHistoryFile) 
  return loss, grads
end

local function validateForIter(iter)
  local loss, dice, b = 0, 0, 0
  for batch in iter() do
    local input, target = batch.input:type(type), batch.target:type(type)
    local output = nil
    if opt.useLocation then
      local location = batch.location:type(type)
      output = net:forward({input, location})
    else
      output = net:forward(input)
    end
    local targetView = target:view(target:nElement())
    dice = dice + diceCoefFromNetOutput(output, targetView)
    loss = loss + criterion:forward(output, targetView)
    b = b + 1
  end
  return loss, dice, b
end

local function writeToValidateLog(loss, dice, b)
  local validateFile = io.open(string.format("%s_val.txt", opt.historyFilePrefix), 'a')
  local toLog = string.format("%d %0.3f %0.3f\n", currentIter, loss/b, dice/b)
  --print(toLog)
  validateFile:write(toLog)
  io.close(validateFile)
end

local function validate()
   --local validateIter = loader:iterator("validate", {augment = true, classId = 2, highResLabel=opt.highRes})
  local nSample = 100
  local validateIter = loader:iterator("validate", {addControl = opt.includeControl, augment = false, classId = opt.targetLabel})
  local loss, dice, b = validateForIter(validateIter)
  writeToValidateLog(loss, dice, b)
end

local function validatePatch()
  local loss, dice, b = 0, 0, 0
  for i=1, nFiles.validate do
    local fileName = string.format("%s/validate/%d.t7", opt.fullSizeDataDir, i)
    local data = torch.load(fileName)
    local validateIter = loader:iteratorRandomPatch(data, {augment = false, classId = opt.targetLabel, patchSize = opt.patchSize})
    local iloss, idice, ib = validateForIter(validateIter)
    loss = loss + iloss
    dice = dice + idice
    b = b + ib
  end
  writeToValidateLog(loss, dice, b)
end

local function checkAndSaveMetrics()
  if currentIter % opt.trainAverageEvery == 0 then
    local trainFile = io.open(string.format("%s_train.txt", opt.historyFilePrefix), 'a')
    local toLog = string.format("%d %0.3f %0.3f\n", currentIter, torch.Tensor(trainLoss):mean(), torch.Tensor(trainDC):mean())
    --print(toLog)
    trainFile:write(toLog)
    io.close(trainFile)
    trainLoss = {}
    trainDC = {}
  end
end

local function saveCheckpoint(epoch, loss)
  local checkpoint = {}
  checkpoint.epoch = epoch
  checkpoint.iter = currentIter
  checkpoint.loss = loss[1]
  checkpoint.optimOpt = optimOpt
  net:clearState()
  checkpoint.model = net
  local fileName = string.format("%sepoch_%d_iter_%d.t7",opt.checkpointDir, epoch, currentIter)
  torch.save(fileName, checkpoint)
end

local function trainWholeImg()
  --local trainIter = loader:iterator("train", {augment = true, classId = 2, highResLabel=opt.highRes})
  local trainIter = loader:iterator("train", {addControl = opt.includeControl, augment = true, classId = opt.targetLabel})
  for epoch = 1+checkpointEpoch, opt.maxEpoch+checkpointEpoch do
    local loss = nil
    for batchData in trainIter() do
      currentIter = currentIter + 1
      sample = batchData
      --_, loss = optim.adagrad(feval, params, optimOpt)
      _, loss = optim.adam(feval, params, optimOpt)
      --print("iter=", currentIter, " loss=", loss[1])
      if currentIter % opt.validateEvery == 0 then
        validate()
      end
      checkAndSaveMetrics()
      collectgarbage()
    end
    if epoch < 100 or epoch % opt.saveModelEvery == 0 then
      saveCheckpoint(epoch, loss)
    end
  end
end

local function trainPatch()
  for epoch = 1+checkpointEpoch, opt.maxEpoch+checkpointEpoch do
    local loss = nil
    for i = 1, nFiles.train do
      local fileName = string.format("%s/train/%d.t7", opt.fullSizeDataDir, i)
      local data = torch.load(fileName)
      local trainIter = loader:iteratorRandomPatch(data, {augment=false, classId = opt.targetLabel, patchSize = opt.patchSize})
      for batchData in trainIter() do
        currentIter = currentIter + 1
        sample = batchData
        _, loss = optim.adam(feval, params, optimOpt)
        if currentIter % opt.validateEvery == 0 then
          validatePatch()
        end
        checkAndSaveMetrics()
        collectgarbage()        
      end
    end
    if epoch < 100 or epoch % opt.saveModelEvery == 0 then
      saveCheckpoint(epoch, loss)
    end
  end
end

if opt.trainPatch then
  trainPatch()
else
  trainWholeImg()
end
