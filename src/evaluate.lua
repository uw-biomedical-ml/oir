require 'nn'
require 'nngraph'
require 'gnuplot'
require 'lfs'
require 'src/model'
require 'cunn'
require 'cutorch'

cutorch.setDevice(2) -- note +1 to make it 0 indexed! sigh lua
cutorch.manualSeed(123)

local cmd = torch.CmdLine()
cmd:option('--dataDir', '/home/saxiao/oir/data/res256/', 'data directory')
cmd:option('--checkpointDir', '/home/saxiao/oir/checkpoint/res256/augment/online/yellow/control/control_0.5/', 'checkpoint directory')
cmd:option('--startEpoch', -1, 'start epoch')
cmd:option('--endEpoch', -1, 'end epoch')
cmd:option('--resultFile', '/home/saxiao/oir/evalResultControlYellow.txt', 'file to store evaluate result')
cmd:option('--writeMode', 'w', 'write mode')
cmd:option('--batchSize', 16, 'batch size')
cmd:option('--nSamples', 100, 'number of samples to draw for evaluation')
cmd:option('--classId', 1, 'classId to classify')
local opt = cmd:parse(arg)

local utils = require 'src/utils'

local Loader = require 'src/OnlineLoader'
local loader = Loader.create(opt)

local criterion = nn.CrossEntropyCriterion()

local function getCheckpointFiles()
  if opt.startEpoch > 0 and opt.endEpoch < 0 then
    return {file = string.format("%sepoch_%d.t7", opt.startEpoch), epoch = opt.startEpoch}
  end
  assert((opt.startEpoch > 0 and opt.endEpoch > 0) or (opt.startEpoch < 0 and opt.endEpoch < 0), "wrong startEpoch and endEpoch options")

  local function findEpoch(str)
    --local i, j = str:find("/[^/]*$")
    return tonumber(str:sub(string.len("epoch_")+1, string.len(str)-3))
  end
  local files = {}
  for file in lfs.dir(opt.checkpointDir) do
    if lfs.attributes(opt.checkpointDir .. file, "mode") == "file" then
      local epoch = findEpoch(file)
      if opt.startEpoch < 0 or (epoch >= opt.startEpoch and epoch <= opt.endEpoch) then
        table.insert(files, {file = opt.checkpointDir .. file, epoch = epoch})
      end
    end
  end

  table.sort(files, function (a,b) return a.epoch < b.epoch end)

  return files
end

local function doEval(checkpointFiles, split)
  print(split)
  local dataList = torch.randperm(opt.nSamples):long()
  local iter = loader:iterator(split, {classId = opt.classId, list = dataList})
  
  local nfiles = #checkpointFiles
  local dice = torch.Tensor(nfiles, opt.nSamples)
  local loss = torch.Tensor(nfiles, math.ceil(opt.nSamples/opt.batchSize))
  for i, f in pairs(checkpointFiles) do
    print(f.epoch)
    local checkpoint = torch.load(f.file)
    local net = checkpoint.model
    if dice:type() ~= net:type() then 
      dice = dice:type(net:type()) 
      loss = loss:type(net:type())
      criterion = criterion:type(net:type())
    end
    local b = 0
    local diceEpoch = nil
    for data in iter() do
      b = b + 1
      local input = data.input:type(net:type())
      local target = data.target:type(net:type())
      local batchSize = input:size(1)
      local output = net:forward(input)
      loss[i][b] = criterion:forward(output, target:view(target:nElement()))
      local _, predict = output:max(2)
      predict = predict:squeeze()
      predict = predict:view(batchSize, -1)
      local diceBatch = utils.diceCoef(predict, target:view(batchSize, -1), 1, true)
      if not diceEpoch then
        diceEpoch = diceBatch
      else
        diceEpoch = torch.cat(diceEpoch, diceBatch)
      end 
    end
    print(string.format("dc = %.3f", diceEpoch:mean()))
    --print(dice:size(), diceEpoch:size())
    dice[i] = diceEpoch[{{1, opt.nSamples}}]
  end
  return dice:mean(2):view(nfiles), dice:var(2):sqrt():view(nfiles), loss:mean(2):view(nfiles)
end

local checkpointFiles = getCheckpointFiles()
local trainDiceMean, trainDiceStd, trainLoss = doEval(checkpointFiles, "train")
local validateDiceMean, validateDiceStd, validateLoss = doEval(checkpointFiles, "validate")
local testDiceMean, testDiceStd, testLoss = doEval(checkpointFiles, "test")

local file = io.open(opt.resultFile, opt.writeMode)
for i, checkpoint in pairs(checkpointFiles) do
  file:write(string.format('%d %f %f %f %f %f %f %f %f %f\n', checkpoint.epoch, trainDiceMean[i], trainDiceStd[i], validateDiceMean[i], validateDiceStd[i], testDiceMean[i], testDiceStd[i], trainLoss[i], validateLoss[i], testLoss[i]))
end
io.close(file)
