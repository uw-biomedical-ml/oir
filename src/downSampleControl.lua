require 'torch'
require 'image'
require 'lfs'
local tnt = require 'torchnet'
local utils = require 'utils'
local imageW, imageH = 256, 256
local nThread = 3

local dataDir = "/data/oir/normal_control"
local files = {}
local function addFiles(dir)
  for fileName in lfs.dir(dir) do
    if fileName ~= '.' and fileName ~= '..' then
      local path = string.format("%s/%s", dir, fileName)
      if lfs.attributes(path, 'mode') == 'directory' then
        addFiles(path)
      else
        table.insert(files, path)
      end
    end
  end
end
addFiles(dataDir)

local function downSampleIterator()
  return tnt.ParallelDatasetIterator{
    nthread = nThread,
    init = function()
      require 'torchnet'
      require 'image'
      gm = require 'graphicsmagick'
    end,
    closure = function()
      return tnt.ListDataset{
          list = torch.range(1, #files):long(),
          load = function(idx)
            local img = gm.Image(files[idx]):toTensor('byte','RGB','DHW')
            local dImg = image.scale(img[1], imageW, imageH)
            return {input = dImg, target = dImg.new():resizeAs(dImg):zero(), rawFilePath = files[idx], labelFilePath = files[idx], originalSize = {w = img:size(2), h = img:size(3)}}
          end,
      }
    end,
  }
end

local iter = downSampleIterator()
local data = {}
local i = 0
for sample in iter() do
  i = i + 1
  local fileName = "/home/saxiao/tmp/control/" .. i .. ".png"
  utils.drawImage(fileName, sample.input)
  table.insert(data, sample)
end
local outputFile = "/home/saxiao/oir/data/res256/control.t7"
torch.save(outputFile, data)
