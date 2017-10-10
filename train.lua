-- rewrite with torchnet OptimEngine
require 'torch'
require 'nn'
require 'optim'
require 'image'

local model = require 'src/model'
local resolution = "res256"
local rootDir = "./"
local modelId = "retina"
local checkpointEpoch, checkpointIter = 0, 0  -- 221300
local cmd = torch.CmdLine()
-- data options
--cmd:option('--dataDir', string.format("%sdata/%s/", rootDir, resolution), 'data directory')
cmd:option('--dataDir', string.format("%sdata/retina/", rootDir), 'data directory')
cmd:option('--batchSize', 32, 'batch size')
cmd:option('--targetLabel', 1, 'target label, retina or yellow is 1, red is 2')
cmd:option('--retina', true, 'training the model for retina')
cmd:option('--nThread', 1, 'number of threads the data loader uses')

cmd:option('--highRes', '/home/saxiao/oir/data/res2048/', 'high resolution label directory')

cmd:option('--trainPatch', false, 'train by randomly selecting a patch from the original image')
cmd:option('--fullSizeDataDir', '/home/saxiao/oir/data/fullres', 'full size data directory')
cmd:option('--patchSize', 256, 'size of the patch')

cmd:option('--includeControl', false, 'including the control images')

-- model options
cmd:option('--nClasses', 2, 'number of classes')
cmd:option('--useLocation', false, 'add location in the input, true only when trainPatch is true')
--cmd:option('--model', '/home/saxiao/oir/checkpoint/red/' .. modelId .. '/epoch_' .. checkpointEpoch .. '_iter_' .. checkpointIter .. '.t7', 'a checkpoint file')

-- training options
cmd:option('--maxEpoch', 500, 'maxumum epochs to train')
cmd:option('--learningRate', 1e-2, 'starting learning rate')
cmd:option('--minLearningRate', 1e-7, 'minimum learning rate')
cmd:option('--momentum', 0.9, 'patch size')
cmd:option('--learningDecayRate', 0.01, 'learning rate decay rate')

-- gpu options
cmd:option('--gpuid', 0, 'patch size')
cmd:option('--seed', 123, 'patch size')

-- checkpoint options
cmd:option('--checkpointDir', rootDir .. "checkpoint/" .. modelId .. "/", 'checkpoint directory')
cmd:option('--saveModelEvery', 10, 'save model every n epochs')
cmd:option('--historyFilePrefix', rootDir .. modelId, 'prefix of the file to save the loss and accuracy for each iteration while training')
cmd:option('--validateEvery', 100, 'run validation every n iterations')
cmd:option('--trainAverageEvery', 50, 'average training metric every n iterations')

local opt = cmd:parse(arg)

local nFiles = {train=682, validate=171, test=214}

paths.mkdir(opt.checkpointDir)

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

local Loader = require 'src/OnlineLoader'
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
    if opt.targetLabel == 2 then
      net = model.uNet1For512(opt)
    else
      net = model.uNet1(opt)
    end
  end
  optimOpt = {learningRate = opt.learningRate}
end

-- TODO: use cross validation to determine?
--local classWeight = torch.Tensor({0.2,0.8})
local criterion = nn.CrossEntropyCriterion()

-- ship the model to the GPU if desired
if opt.gpuid > -1 then
  net = net:cuda()
  criterion = criterion:cuda()
end

local params, grads = net:getParameters()
local type = net:type()

local currentIter = 0 + checkpointIter
local function calHits(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)
  local hit = torch.eq(predict, target):sum()
  return hit
end

-- N classes, not including background
local function diceCoef(predict, label)
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
    a = a + pt
    b = b + psum
    c = c + tsum
  end
  local dice = (2*a + eps)/(b + c + eps)
  return dice, eachDice
end

local function diceCoefFromNetOutput(output, target)
  local _, predict = output:max(2)
  predict = predict:squeeze():type(type)  -- 1 is normal, 2 is target to predict
  return diceCoef(predict, target)
end

local sample = nil
local trainLoss = {}
local trainDC = {}
local feval = function(w)
  if w ~= params then
    params:copy(w)
  end
  grads:zero()
  
  local data, label, location = sample.input, sample.target, sample.location
  local originalType = data:type()
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
  
  print(string.format("%d, loss=%.3f, dice=%.3f", currentIter, loss, dice))
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
  checkpoint.opt = opt
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
      _, loss = optim.adam(feval, params, optimOpt)
      if currentIter % opt.validateEvery == 0 then
        net:evaluate()  -- this is important as some modules are computed differently in training and test time, e.g. batchNormalization
        validate()
        net:training()
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
          net:evaluate()
          validatePatch()
          net:training()
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

net:training()
if opt.trainPatch then
  trainPatch()
else
  trainWholeImg()
end
