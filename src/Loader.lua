require 'torch'
require 'lfs'

local Loader = {}
Loader.__index = Loader

function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  
  self.dataDir = opt.trainData
  local nfiles = opt.nfiles
  if not nfiles or nfiles < 0 then
    local stats = torch.load(self.dataDir .. "/stats.t7")
    nfiles = stats.nfiles
  end
  self.nfiles = nfiles
--  local all = torch.randperm(nfiles)
  local all = torch.linspace(1, nfiles, nfiles)
  local trainSize = nfiles * opt.trainSize
  self.set = {}
  self.set["train"] = all[{{1, trainSize}}]
  self.set["validate"] = all[{{trainSize + 1, nfiles}}]
  self.batchSize = opt.batchSize
  
  return self
end

function Loader:sample(split, nSample)
  print(split, nSample)
  local data, label = torch.ByteTensor(), torch.ByteTensor()
  local indexes = torch.randperm(nSample)
  for i = 1, nSample do
     local fileName = self.dataDir .. self.set[split][indexes[i]] .. ".t7"
     local fileData = torch.load(fileName)
     if data:nElement() == 0 then
        data:resize(nSample, 1, fileData.raw:size(1), fileData.raw:size(2)):zero()
        label:resize(nSample, fileData.raw:size(1), fileData.raw:size(2))
     end
     data[i][1] = fileData.raw
     label[i] = fileData.label     
  end
  return data, label+1
end

function Loader:iterator(split)
  local it = {}
  local fileCursor = 0
  it.epoch = 0  
  local function getQuadrant(output, input, b)
    local imageW, imageH = input:size(1), input:size(2)
    output[b] = input[{{1, imageW/2},{1, imageH/2}}]
    output[b+1] = input[{{1, imageW/2}, {imageH/2+1, imageH}}]
    output[b+2] = input[{{imageW/2+1, imageW}, {1, imageH/2}}]
    output[b+3] = input[{{imageW/2+1, imageW}, {imageH/2+1, imageH}}]
  end
  it.nextBatch = function ()
    local data = torch.ByteTensor()
    local label = torch.ByteTensor()
    for i = 1, self.batchSize do
      fileCursor = fileCursor + 1
      if fileCursor > self.set[split]:size(1) then
        fileCursor = 1
        it.epoch = it.epoch + 1
      end
      local fileName = self.dataDir .. self.set[split][fileCursor] .. ".t7"
      local fileData = torch.load(fileName)
      if data:nElement() == 0 then
        data:resize(self.batchSize, 1, fileData.raw:size(1), fileData.raw:size(2)):zero()
        label:resize(self.batchSize, fileData.raw:size(1), fileData.raw:size(2))
      end
      data[i][1] = fileData.raw
      label[i] = fileData.label
--      getQuadrant(data[{{},1}], fileData.raw, i)
--      getQuadrant(label, fileData.label, i)
    end
    return data, label+1  -- 1 is normal, 2 is hightlighted/abnormal
  end
  return it
end

return Loader
