require 'image'

local tnt = require 'torchnet'
local utils = require 'utils'

local dataDir = "/home/saxiao/oir/data/res256/"
local nLoaderThread = 3
local batchSize = 4
local var = 0.3

local plotDir = "/home/saxiao/tmp/"
local plotImage = false

local function rotateImage(raw, label, theta)
  local rawRotated = image.rotate(raw, theta, mode)
  local labelRotated = image.rotate(label, theta, mode)
  return {
    raw = rawRotated,
    label = labelRotated,
  }
end

local function varyBrightness(raw)
  local scale = 1 + torch.uniform(-var, var)
  raw = raw:float() 
  return {
    raw = torch.mul(raw, scale),
    alpha = scale,
  }
end

local function varyContrast(raw, scale)
  raw = raw:float()
  local mean = raw[1]:mean()
  return raw:mul(scale):add((1 - scale) * mean)
end

local function transform(sample, var)
      local theta = torch.uniform(0, 2 * math.pi)
      local transformed = rotateImage(sample.raw, sample.label, theta)
      local scale = 1 + torch.uniform(-var, var)
      print(scale, transformed.raw[1]:min(), transformed.raw[1]:max())
      transformed.raw = varyContrast(transformed.raw, scale)
      print(scale, transformed.raw:min(), transformed.raw:max())

      if plotImage then
        local rawFileName = string.format("%s%d_raw.png", plotDir, sample.idx)
        utils.drawImage(rawFileName, sample.raw[1])
        local fileName = string.format("%s%d_%0.3f_%.3f.png", plotDir, sample.idx, theta, scale)
        utils.drawImage(fileName, transformed.raw[1]:byte())
      end

      return {
        input = transformed.raw,
        target = transformed.label,
        idx = sample.idx,
      }
end

--local d = tnt.TableDataset{data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
local d = torch.rand(10,2)

local function iterator(split)
  local dataSet = torch.load(string.format("%s%s.t7", dataDir, split))
  return tnt.ParallelDatasetIterator{
    nthread = nLoaderThread,
    init = function() 
      require 'torchnet'
      require 'image'
    end,
    closure = function()
      return tnt.BatchDataset{
        batchsize = batchSize,
        dataset = tnt.ListDataset{
          list = torch.range(1,10):long(),
          load = function(idx)
            local sample = {}
            sample.raw = dataSet.raw[idx]
            sample.label = dataSet.label[idx]
            sample.idx = idx
            return transform(sample, self.var)
          end
        }
      }
    end,
  }
end

local function varyContrast(input, scale)
  input = input:float()
  local mean = input:mean()
  return input:mul(scale):add((1 - scale) * mean)
end

--local iter = getIterator('train')
--local iter = iterator(dataSet)
local opt = {}
opt.batchSize = 4
opt.var = var
opt.dataDir = dataDir
opt.nThread = 1
local Loader = require 'OnlineLoader'
local loader = Loader.create(opt)
--local iter = loader:iterator("test", {ordered = true, augment = false, list = torch.range(1,10):long(), classId=2, highResLabel="/home/saxiao/oir/data/res2048/"})
--local iter = loader:iterator("test", {augment = false, ordered = true, classId=1})
opt.fullSizeDataDir = "/home/saxiao/oir/data/fullres"
opt.nFiles = {train=682}
opt.classId = 2
opt.augment = true
print(opt)
local iter = loader:iteratorRandomPatch("train", opt)
--for epoch = 1, 2 do
--print(string.format("epoch = %d", epoch))
local b = 0
for batch in iter() do 
  b= b + 1
  print("location", batch.location:size())
  print(batch.location)
  print(string.format("batch cnt: %d, label min: %d, label max: %d",batch.target:size(1), batch.target:min(), batch.target:max()))
  for i = 1, batch.target:size(1) do
    print(i, batch.idx[i])
    --utils.drawImage(string.format("/home/saxiao/tmp/oirpatch/%d_raw.png", batch.idx[i]), batch.input[i][1])
    --utils.drawImage(string.format("/home/saxiao/tmp/oirpatch/%d_label.png", batch.idx[i]), batch.input[i][1], batch.target[i])
    if drawRetina then
      local retinaMask = batch.input[i][1]:new():resizeAs(batch.input[i][1]):copy(batch.input[i][1])
      retinaMask = varyContrast(retinaMask, 1.5)
      local retina = retinaMask.new():resizeAs(retinaMask):zero()
      retina:maskedFill(retinaMask:gt(10), 230)
      utils.drawImage(string.format("/home/saxiao/tmp/retina/%d_retina.png", batch.idx[i]), retina)
    end
    --local dlabel = batch.input[i].new():resizeAs(batch.input[i]):zero()
    --local imask = image.scale(batch.target[i]-1, 256, 256)
    --dlabel:maskedFill(imask:gt(0.5), 1)
--    local dlabel = batch.target[i]-1
--    utils.drawImage(string.format("/home/saxiao/tmp/%d.png", i), batch.input[i][1], dlabel)
  end
    --print(torch.type(sample))
  --print(sample.input:size(), sample.target:size())
  --print(sample.target:min(), sample.target:max())
  if b > 2 then break end
end
--end
--while epoch < maxEpoch do
--for sample in iter() do
--  cnt = cnt + sample.input:size(1)
--  print("cnt = ", cnt)
--  print(sample.target:min(), sample.target:max())
--end
--epoch = epoch + 1
--end
