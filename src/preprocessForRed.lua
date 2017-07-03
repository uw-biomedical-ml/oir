--require 'cunn'
--require 'cutorch'
--cutorch.setDevice(1) -- note +1 to make it 0 indexed! sigh lua
--cutorch.manualSeed(123)
--require 'model'
local utils = require 'utils'

local dataDir = "/home/saxiao/oir/data/fullres/test"
--local checkpoint = torch.load("/home/saxiao/oir/checkpoint/res256/augment/online/yellow/epoch_150.t7")
--local model = checkpoint.net:cuda()
local N = 214
local downsampleSize = 256

local coords = {}
local function getCoordTensor(size)
  if not coords[size] then
    local t = torch.LongTensor(size, size)
    for h=1, size do
      for w=1, size do
        t[h][w] = size*(h-1) + w
      end
    end
    coords[size] = t
  end
  return coords[size]
end

for i=1, N do
  print(i)
  local fileName = string.format("%s/%d.t7", dataDir, i)
  local file = torch.load(fileName)
  local input, target = file.input, file.target
  --local coordTensor = getCoordTensor(target:size(1))
  local dinput, dtarget = utils.scale(input, target, downsampleSize, downsampleSize)
  local retinaLabel = utils.learnByKmeansThreshold(dinput, {useLog = true})
  local _, orignalRetinaLabel = utils.scale(nil, retinaLabel, target:size(1), target:size(2))
  --local retinaCoord = coordTensor:maskedSelect(orignalRetinaLabel)
  --local redCoord = coordTensor:maskedSelect(target:eq(2))
  file.retinaLabel = orignalRetinaLabel
  --file.retinaCoord = retinaCoord
  --file.redCoord = redCoord
  torch.save(fileName, file) 
end

