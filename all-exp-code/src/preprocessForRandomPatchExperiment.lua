local utils = require 'utils'

local dataDir = "/home/saxiao/oir/data/fullres/test"
local N = 214
local downsampleSize = 256

local function addRetinaLabel()
  for i=1, N do
    print(i)
    local fileName = string.format("%s/%d.t7", dataDir, i)
    local file = torch.load(fileName)
    local input, target = file.input, file.target
    local dinput, dtarget = utils.scale(input, target, downsampleSize, downsampleSize)
    local retinaLabel = utils.learnByKmeansThreshold(dinput, {useLog = true})
    local _, orignalRetinaLabel = utils.scale(nil, retinaLabel, target:size(1), target:size(2))
    file.retinaLabel = orignalRetinaLabel
    torch.save(fileName, file) 
  end
end

addRetinaLabel()

