require 'torch'
require 'lfs'

local Loader = {}
Loader.__index = Loader

function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  
  self.trainData= opt.trainData
  local nfiles = opt.nfiles
  if not nfiles or nfiles < 0 then
    local stats = torch.load(self.trainData .. "/stats.t7")
    nfiles = stats.nfiles
  end
   
  self.originalRawPaths = {train = torch.load("/home/saxiao/oir/data/res256/train/rawPath.t7"), test = torch.load("/home/saxiao/oir/data/res256/test/rawPath.t7")}
  self.originalLabelPaths = {train = torch.load("/home/saxiao/oir/data/res256/train/labelPath.t7"), test = torch.load("/home/saxiao/oir/data/res256/test/labelPath.t7")}
  
  self.nfiles = nfiles
  local all = torch.linspace(1, nfiles, nfiles)
  local trainSize = nfiles * opt.trainSize
  self.set = {}
  self.set["train"] = all[{{1, trainSize}}]
  self.set["validate"] = all[{{trainSize + 1, nfiles}}]
  self.batchSize = opt.batchSize
 
  self.testData = opt.testData
  local testStats = torch.load(self.testData .. "/stats.t7")
  self.ntestFiles = testStats.nfiles
  self.set["test"] = torch.linspace(1, self.ntestFiles, self.ntestFiles)  
  return self
end

function Loader:sample(split, nSample)
  print(split, nSample)
  local dataSet = self.trainData
  if split == "test" then dataSet = self.testData end
  local pathSplit = split == "test" and "test" or "train"
  local rawPath, labelPath = self.originalRawPaths[pathSplit], self.originalLabelPaths[pathSplit]
  local data, label = torch.ByteTensor(), torch.ByteTensor()
  local dataFiles = {}
  local indexes = nil
  if nSample < 0 then
    nSample = self.set[split]:size(1)
    indexes = torch.linspace(1, nSample, nSample)
  else
    indexes = torch.randperm(nSample)
  end
  print(split, nSample)
  for i = 1, nSample do
     local fileId = self.set[split][indexes[i]]
     local fileName = dataSet .. fileId .. ".t7"
     local fileData = torch.load(fileName)
     if data:nElement() == 0 then
        data:resize(nSample, 1, fileData.raw:size(1), fileData.raw:size(2)):zero()
        label:resize(nSample, fileData.raw:size(1), fileData.raw:size(2))
     end
     data[i][1] = fileData.raw
     label[i] = fileData.label
     local pathIdx = split == "test" and fileId or math.ceil(fileId / 11)
     dataFiles[i] = {raw = rawPath[pathIdx], label = labelPath[pathIdx]}
  end
  return data, label+1, dataFiles
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
      local fileName = self.trainData .. self.set[split][fileCursor] .. ".t7"
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
