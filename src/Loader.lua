require 'torch'
require 'lfs'
require 'gnuplot'
require 'paths'

local gm = require 'graphicsmagick'

local Loader = {}
Loader.__index = Loader

local function getAllPaths(dir, labeled)
  lfs.chdir(dir)

  for file in lfs.dir(dir) do
    if file ~= "." and file ~= ".." then
      local path = dir .. "/" .. file
      if lfs.attributes(path, "mode") == "directory" then
        getAllPaths(path, labeled) 
      elseif lfs.attributes(path, "mode") == "file" and string.match(file, "quantified") then
        table.insert(labeled,path)
      end
    end
  end
end

local function getRawFilePath(labeledFile)
  local qStart = string.find(labeledFile, "quantified")
  local rawFile = nil
  if string.sub(qStart-1, qStart-1) ~= " " then
    rawFile = string.sub(labeledFile,1,qStart-1) .. ".tif"
  else
    rawFile = string.sub(labeledFile,1,qStart-2) .. ".tif"
  end
  return rawFile
end

local function retrieveAndSaveAllPaths(dir)
  local pathsFile = dir .. "allPaths.t7"
  local cleanedLabeled = nil
  if not paths.filep(pathsFile) then
      local labeled = {}
      getAllPaths(dir, labeled)
      cleanedLabeled = {}
      for file in labeled do
        local rawFile = getRawFilePath(file)
        if paths.filep(rawFile) then
          table.insert(cleanedLabeled, file)
        end
      end
      torch.save(pathsFile, cleanedLabeled)
  else
    cleanedLabeled = torch.load(pathsFile)
  end
end

function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  
  local allPaths = retrieveAndSaveAllPaths(opt.dir)
  local trainSize = #allPaths * opt.trainSize
  local validateSize = #allPaths * opt.validateSize
  local trainSet, validateSet, testSet = {}, {}, {}
  local cnt = 1
  for k, v in pairs(allPaths) do
    if cnt <= trainSize then
      table.insert(trainSet, v)
    elseif cnt <= trainSize + validateSize then
      table.insert(validateSet, v)
    else
      table.insert(testSet, v)
    end
  end
  self.allData = {}
  self.allData["train"] = trainSet
  self.allData["validate"] = validateSet
  self.allData["test"] = testSet
  return self
end

function Loader:iterator(split)
  local it = {}
  local fileCursor = 0
  local rawImage, labels = nil, nil
  local patchCenterX, patchCenterY = 0
  
  -- raw images only has data in channel 1
  local function getBatchData()
    local batchSize, patchSize = self.batchSize, self.patchSize
    local imageW, imageH = rawImage:size(2), rawImage:size(3)
    local data = torch.Tensor(batchSize, patchSize, patchSize)
    local label = torch.ones(batchSize)
    
    local cnt = 0
    while cnt < batchSize and patchCenterX <= imageW - patchSize/2 and patchCenterY <= imageH - patchSize/2 do
      cnt = cnt + 1
      data[cnt] = rawImage[1]:sub(patchCenterX-patchSize/2+1, patchCenterX+patchSize/2, patchCenterY-patchSize/2+1, patchCenterY+patchSize/2)
      label[cnt] = labels[patchCenterX][patchCenterY]
      patchCenterX = patchCenterX + self.spacing
      if patchCenterX > imageW - patchSize/2 then
        patchCenterY = patchCenterY + self.spacing
        if patchCenterY <= imageH - patchSize/2 then
          patchCenterX = patchSize/2
        end
      end
    end
    if cnt < batchSize then
      patchCenterX, patchCenterY = 0, 0
    end
    return data, label
  end
  
  local function getLabelImage(labeledFile)
    local labeledImage = gm.Image(labeledFile):toTensor('byte','RGB','DHW')
    labels = torch.eq(rawImage[2], labeledImage[2]) + 1  -- class = 1 is abnormal, class = 2 is normal
  end
  
  it.nextBatch = function()
    if patchCenterX < 1 then
      fileCursor = fileCursor + 1
      if fileCursor > #self.allData[split] then
        fileCursor = 1
      end
      local labeledFile = self.allData[split][fileCursor]
      rawImage = gm.Image(getRawFilePath(labeledFile)):toTensor('byte','RGB','DHW')
      labels = getLabelImage(labeledFile)
      patchCenterX, patchCenterY = self.patchSize / 2, self.patchSize / 2
    end
    return getBatchData()
  end
  return it
end

return Loader

