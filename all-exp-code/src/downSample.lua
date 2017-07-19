-- TODO: rewrite with torchnet parallel data iterator
require 'torch'
require 'image'

local gm = require 'graphicsmagick'
local utils = require 'utils'
local imageW, imageH = 256, 256


local function loadFile(txtFile)
  local t = {}
  local cnt = 0
  local file = io.open(txtFile)
  if file then
    for line in file:lines() do
      table.insert(t, line)
      cnt = cnt + 1
    end
  end
  return t, cnt
end


local function downSampling(raw, label, w, h, imageFileName)
  local draw = image.scale(raw, w, h)
  local labelDS = image.scale(label, w, h)
  local dlabel = labelDS.new():resizeAs(labelDS):zero()
  dlabel:maskedFill(labelDS:gt(0.5), 1)
  dlabel:maskedFill(labelDS:gt(1.5), 2)
  if imageFileName then
    utils.drawImage(imageFileName, draw, dlabel)
  end

  return draw, dlabel
end

local function generateData(inputDir, outputDir, split)
  local rawPathFile = torch.load(string.format("%s%s_raw.t7", inputDir, split))
  local labelPathFile = torch.load(string.format("%s%s_label.t7", inputDir, split))
  local data = {}
--  local rawImages, labels = {}, {}
--  local rawFilePath, labelFilePath, originalSize = {}, {}, {}
  local cnt, redCnt = 0, 0
  for i, rawPath in pairs(rawPathFile) do
    local rawImage = gm.Image(rawPath):toTensor('byte','RGB','DHW')
    local labelImage = gm.Image(labelPathFile[i]):toTensor('byte','RGB','DHW')
    if rawImage:size(2) == labelImage:size(2) then
      cnt = cnt + 1
      print(cnt)
      local label = utils.getLabel(rawImage, labelImage) -- bg/normal = 1, yellow = 1, red = 2
      if label:max() == 2 then
        redCnt = redCnt + 1
      end
      local imageFileName = string.format("%splot/%d.png", outputDir, cnt)
      local draw, dlabel = downSampling(rawImage[1], label, imageW, imageH)
      data[i] = {input = draw, target = dlabel, rawFilePath = rawPath, labelFilePath = labelPathFile[i], originalSize = {w = label:size(1), h = label:size(2)}}
    end
    if cnt > 10 then break end
  end
  
--  local rawData = torch.ByteTensor(#rawImages, 1, imageW, imageH)
--  local labelData = torch.ByteTensor(#rawImages, imageW, imageH)
--  for i, v in pairs(rawImages) do
--    rawData[i][1] = v
--    labelData[i] = labels[i]
--  end
--  local data = {raw = rawData, label = labelData, rawFilePath = rawFilePath, labelFilePath = labelFilePath, originalSize = originalSize}
  torch.save(string.format("%sres%d/%s.t7", outputDir, imageW, split), data)
  print(string.format("%s set has total %d, redCnt: %d", split, cnt, redCnt))
end

local dir = "/home/saxiao/oir/data/"
generateData(dir, dir, "train")
generateData(dir, dir, "validate")
generateData(dir, dir, "test")

