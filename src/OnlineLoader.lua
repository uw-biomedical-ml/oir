local tnt = require 'torchnet'

local Loader = {}
Loader.__index = Loader

function Loader.create(opt)
  local self = {}
  setmetatable(self, Loader)
  self.dataDir = opt.dataDir
  self.nThread = opt.nThread or 1
  self.batchSize = opt.batchSize or 32
  self.var = opt.var or 0.3
  if opt.includeControl then  
    self.controlData = torch.load(string.format("%scontrol.t7", self.dataDir))
    local list = torch.range(1, #self.controlData)  -- 24
    self.controlDataList = {train=list[{{1,16}}], validate=list[{{17,20}}], test=list[{{21,24}}]}
  end
  self.coords = {}
  return self
end

local function rotateImage(theta, input, target)
  local inputRotated = image.rotate(input, theta, mode)
  local targetRotated = nil
  if target then
    targetRotated = image.rotate(target, theta, mode)
  end
  return {
    input = inputRotated,
    target = targetRotated,
  }
end

local function varyContrast(input, scale)
  input = input:float()
  local mean = input:mean()
  return input:mul(scale):add((1 - scale) * mean)
end

local function transform(sample, var, plotOpt)
      local theta = torch.uniform(0, 2 * math.pi)
      local transformed = rotateImage(theta, sample.input, sample.target)
      local contrastVarUp = var
      if plotOpt then contrastVarUp = 0 end
      local scale = 1 + torch.uniform(-var, contrastVarUp)
      transformed.input = varyContrast(transformed.input, scale)

      return {
        input = transformed.input,
        target = transformed.target,
      }
end

local function getCoords(coords, size)
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

local function selectRandomCoord(list, H)
  local id = torch.random(1, list:size(1))
  return math.floor(id/H)+1, id%H
end

local function calStartPoint(center, L, size)
  local pstart = center - math.floor(L/2)
  if pstart < 1 then
    pstart = 1
  end
  if pstart + L -1 > size then
    pstart = size - L + 1
  end
  return pstart
end

local function normalize(x, size)
  return (x / size) * 2 - 1
end

function Loader:iteratorRandomPatch(data, opt)
  --local dataDir = string.format("%s/%s", opt.fullSizeDataDir, split)
  --local nFiles = opt.nFiles[split]
  local rawImage = data.input
  local label = data.target
  local coords = getCoords(self.coords, rawImage:size(1))
  local nPatches = opt.nPatches and opt.nPatches or 64
  return tnt.ParallelDatasetIterator{
    nthread = self.nThread,
    ordered = opt.ordered and opt.ordered or false,
    init = function()
      require 'torchnet'
      require 'image'
    end,
    closure = function()
      --local list = torch.randperm(nFiles):long()
      local list = torch.range(1, nPatches):long()
      local patchSize = opt.parchSize and opt.patchSize or 256
      --local margin = opt.margin and opt.margin or 50
      return tnt.BatchDataset{   
        batchsize = self.batchSize,
        dataset = tnt.ListDataset{
          list = list,
          load = function(idx)
            --local data = torch.load(string.format("%s/%d.t7", dataDir, idx))
            --local rawImage = data.input
            --local label = data.target
            local category = math.random()
            --local coords = getCoords(self.coords, rawImage:size(1))
            local selectFrom = nil
            if category < 0.5 then
              -- pick one from the reds 
              selectFrom = coords:maskedSelect(label:eq(opt.classId))
            else
              -- pick one from the retina area
              selectFrom = coords:maskedSelect(data.retinaLabel)
            end
            local ch, cw = selectRandomCoord(selectFrom, label:size(1))
            --print(ch, cw)
            local hstart= calStartPoint(ch, patchSize, label:size(1))
            local wstart = calStartPoint(cw, patchSize, label:size(2)) 
            local sample = {}
            sample.input = rawImage:sub(hstart, hstart+patchSize-1, wstart, wstart+patchSize-1)
            sample.target = label:sub(hstart, hstart+patchSize-1, wstart, wstart+patchSize-1)
            if opt.augment then
              local transformed = transform(sample, self.var, opt.plotOpt)
              sample.input:copy(transformed.input)
              sample.target:copy(transformed.target)
            end
            local input = sample.input.new():resize(1, sample.input:size(1), sample.input:size(2))
            input[1]:copy(sample.input)
            sample.input = input
            local target = sample.target.new():resize(sample.target:size(1), sample.target:size(2))
            target:copy(sample.target)
            sample.target = target
            if opt.classId then
              sample.target = sample.target:eq(opt.classId)
            end
            sample.target:add(1)
            sample.originalSize = data.originalSize
            sample.rawFilePath = data.rawFilePath
            sample.labelFilePath = data.labelFilePath
            sample.idx = idx
            sample.location = torch.Tensor{normalize(ch,rawImage:size(1)), normalize(cw,rawImage:size(2))}
            return sample
          end
        }
      }
    end,
  }
end


function Loader:iterator(split, opt)
  --print("classId", opt.classId)
  --print("ordered", opt.ordered)
  local dataSet = torch.load(string.format("%s%s.t7", self.dataDir, split))
  --print(string.format("highResLabel: %s", opt.highResLabel))
  return tnt.ParallelDatasetIterator{
    nthread = self.nThread,
    ordered = opt.ordered and opt.ordered or false,
    init = function()
      require 'torchnet'
      require 'image'
    end,
    closure = function()
      local list = torch.randperm(#dataSet):long()
      if opt.shuffleOff then list = torch.range(1, #dataSet):long() end
      if opt.list then list = opt.list end 
      if opt.nSample and opt.nSample < list:size(1) then
        list = list[{{1, opt.nSample}}]
      end
      return tnt.BatchDataset{   
        batchsize = self.batchSize,
        dataset = tnt.ListDataset{
          list = list,
          load = function(idx)
            local sample = {}
            sample.input = dataSet[idx].input.new():resizeAs(dataSet[idx].input):copy(dataSet[idx].input)
            sample.target = dataSet[idx].target.new():resizeAs(dataSet[idx].target):copy(dataSet[idx].target)
            if opt.highResLabel then
              local highRes = torch.load(string.format("%s%s/%d.t7", opt.highResLabel, split, idx))
              sample.target:resizeAs(highRes.target)
              sample.target:copy(highRes.target)
            end
            if opt.augment then
              local transformed = transform(sample, self.var, opt.plotOpt)
              sample.input:copy(transformed.input)
              sample.target:copy(transformed.target)
            end
            sample.input = sample.input:view(1, sample.input:size(1), sample.input:size(2))
            if opt.classId then
              sample.target = sample.target:eq(opt.classId)
            end
            sample.target:add(1)
            sample.rawFilePath = dataSet[idx].rawFilePath
            sample.labelFilePath = dataSet[idx].labelFilePath
            sample.originalSize = dataSet[idx].originalSize
            sample.idx = idx
            return sample
          end
        }
      }
    end,
    transform = function(batch)
      if opt.addControl and split ~= 'control' then
      local controlDataList = self.controlDataList[split]
        local list = torch.randperm(controlDataList:size(1))
        if split == 'train' then
          local controlRatio = 1/2  -- 1/6
          if math.ceil(batch.input:size(1)*controlRatio) < list:size(1) then
            list = list[{{1, math.ceil(batch.input:size(1)*controlRatio)}}]
          end
        end
        local control = batch.input.new():resize(list:size(1),batch.input:size(2), batch.input:size(3), batch.input:size(4))
        for i = 1, list:size(1) do
          local data = self.controlData[controlDataList[list[i]]]
          local sample = {}
          sample.input = data.input.new():resizeAs(data.input):copy(data.input)
          local transformed = transform(sample, self.var, opt.plotOpt)
          control[i][1]:copy(transformed.input) 
          table.insert(batch.rawFilePath, data.rawFilePath)
          table.insert(batch.labelFilePath, data.labelFilePath)
          table.insert(batch.originalSize, data.originalSize)
        end
        local controlTarget = control.new():resize(list:size(1),batch.input:size(3), batch.input:size(4)):fill(1)
        batch.input = torch.cat(batch.input, control, 1)
        batch.target = torch.cat(batch.target, controlTarget, 1)
      end
      return batch
    end,
  }
end

return Loader
