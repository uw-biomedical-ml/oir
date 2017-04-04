require 'torch'
require 'lfs'
require 'gnuplot'
require 'paths'
require 'image'

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
  local rawFile = string.sub(labeledFile,1,qStart-1)
  rawFile = string.gsub(rawFile, "%s*$", "") .. ".tif"
  return rawFile
end

local function retrieveAndSaveAllPaths(pathsFile, dir) 
  local cleanedLabeled = nil
  if not paths.filep(pathsFile) then
      local labeled = {}
      getAllPaths(dir, labeled)
      cleanedLabeled = {}
      for _, file in pairs(labeled) do
        local rawFile = getRawFilePath(file)
        if paths.filep(rawFile) then
          table.insert(cleanedLabeled, file)
        end
      end
      torch.save(pathsFile, cleanedLabeled)
  else
    cleanedLabeled = torch.load(pathsFile)
  end
  return cleanedLabeled
end

function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  self.batchSize = opt.batchSize
  -- for the first iterator
  self.patchSize = opt.patchSize
  local spacing = opt.spacing
  if not spacing or spacing < 0 then
    spacing = opt.patchSize
  end
  self.spacing = spacing  
  
  -- for the downSampled iterator
  self.imageW = opt.imageW
  self.imageH = opt.imageH
 
  local allPaths = retrieveAndSaveAllPaths(opt.pathsFile, opt.dataDir)
  local trainSize = table.getn(allPaths) * opt.trainSize
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
    cnt = cnt + 1
  end
  self.allData = {}
  self.allData["train"] = trainSet
  self.allData["validate"] = validateSet
  self.allData["test"] = testSet
  return self
end

local function getLabel(labeledFile, rawImage)
    local labeledImage = gm.Image(labeledFile):toTensor('byte','RGB','DHW')
    local labels = rawImage[2]:eq(labeledImage[2]) + 1 -- hightlighted (abnormal) is 1, normal + background is 2
    return labels 
end

function Loader:iterator(split)
  local it = {}
  local fileCursor = 0
  local rawImage, labels = nil, nil
  local patchCenterX, patchCenterY = 0, 0
 
  it.reset = function()
    fileCursor = 0
    patchCenterX, patchCenterY = 0, 0
  end
 
  -- raw images only has data in channel 1
  local function getBatchData()
    local batchSize, patchSize = self.batchSize, self.patchSize
    local imageW, imageH = rawImage:size(2), rawImage:size(3)
    local data = torch.Tensor(batchSize, 1, patchSize, patchSize)
    local label = torch.ones(batchSize, patchSize, patchSize)
    
    local cnt = 0
    while cnt < batchSize and patchCenterX <= imageW - patchSize/2 and patchCenterY <= imageH - patchSize/2 do
      cnt = cnt + 1
      data[cnt][1] = rawImage[1]:sub(patchCenterX-patchSize/2+1, patchCenterX+patchSize/2, patchCenterY-patchSize/2+1, patchCenterY+patchSize/2)
      label[cnt] = labels:sub(patchCenterX-patchSize/2+1, patchCenterX+patchSize/2, patchCenterY-patchSize/2+1, patchCenterY+patchSize/2)
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
    return data, label:view(label:nElement())
  end
  
--  local function getLabelImage(labeledFile)
--    local labeledImage = gm.Image(labeledFile):toTensor('byte','RGB','DHW')
--    local labels = torch.eq(rawImage[2], labeledImage[2]) + 1  -- class = 1 is abnormal, class = 2 is normal
--    return labels
--  end
 
  it.getImageSize = function()
    return rawImage:size()
  end
  
  it.getFileCursor = function()
    return fileCursor
  end

  it.getPatchCenterX = function()
    return patchCenterX
  end

  it.getPatchCenterY = function()
    return patchCenterY
  end
 
  it.nextBatch = function()
    if patchCenterX < 1 then
      fileCursor = fileCursor + 1
      if fileCursor > #self.allData[split] then
        fileCursor = 1
      end
      local labeledFile = self.allData[split][fileCursor]
      rawImage = gm.Image(getRawFilePath(labeledFile)):toTensor('byte','RGB','DHW')
      labels = getLabel(labeledFile, rawImage)
      patchCenterX, patchCenterY = self.patchSize / 2, self.patchSize / 2
    end
    return getBatchData()
  end
  return it
end

function Loader:iterator2(split)
  local it = {}
  local fileCursor = 0
  
  it.nextBatch = function()
--    local nFiles = self.batchSize / 4
    local b = 1
    local data, label = {}, {} -- raw images may have different size, so cannot use a tensor to store
    for fcnt = 1, self.batchSize do
      fileCursor = fileCursor + 1
      if fileCursor > #self.allData[split] then
        fileCursor = 1
      end
      local labeledFile = self.allData[split][fileCursor]
      local rawImage = gm.Image(getRawFilePath(labeledFile)):toTensor('byte','RGB','DHW')
      local rawLabel = getLabel(labeledFile, rawImage)
      table.insert(data, rawImage[1]) -- only the first channel has useful info
      table.insert(label, rawLabel)  
  end
    return data, label
  end

  return it  
end

function Loader:iteratorDownSampled(split)
  local it = {}
  local fileCursor = 0
     
  local function getDownSampledLabel(labeledFile, rawImage)
    local originLabels = getLabel(labeledFile, rawImage)
    local downSampled = image.scale(originLabels, self.imageW*2, self.imageH*2)
    local labels = torch.gt(originLabels, 1.5)  -- 1 is abnormal, 2 is normal
    return labels
  end

  local function getQuadrant(output, input, b)
    local p1 = input[{{1, self.imageW},{1, self.imageH}}]
    output[b] = input[{{1, self.imageW},{1, self.imageH}}]
    output[b+1] = input[{{self.imageW+1, self.imageW*2}, {1, self.imageH}}]
    output[b+2] = input[{{1, self.imageW}, {self.imageH+1, self.imageH*2}}]
    output[b+3] = input[{{self.imageW+1, self.imageW*2}, {self.imageH+1, self.imageH*2}}]
  end

  it.nextBatch = function()
    local nFiles = self.batchSize / 4
    local fcnt, b = 0, 1
    local data = torch.ByteTensor(self.batchSize, 1, self.imageW, self.imageH)
    local label = torch.ByteTensor(self.batchSize, self.imageW, self.imageH)
    while fcnt < nFiles do
      fileCursor = fileCursor + 1
      if fileCursor > #self.allData[split] then
        fileCursor = 1
      end
      fcnt = fcnt + 1
      local labeledFile = self.allData[split][fileCursor]
      local rawImage = gm.Image(getRawFilePath(labeledFile)):toTensor('byte','RGB','DHW')
      local dataDownSampled = image.scale(rawImage[1], self.imageW*2, self.imageH*2)
      local labelDownSampled = getDownSampledLabel(labeledFile, rawImage)
      getQuadrant(data[{{},1}], dataDownSampled, b)
      getQuadrant(label, labelDownSampled, b)
      b = b + 4
    end
    return data, label:view(label:nElement())
  end
  
  return it
end

return Loader

