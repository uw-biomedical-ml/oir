require 'torch'
require 'image'
local gm = require 'graphicsmagick'

local rawPathFile = torch.load("/home/saxiao/oir/data/res256/test/rawPath.t7")
local labelPathFile = torch.load("/home/saxiao/oir/data/res256/test/labelPath.t7")
local imageW, imageH = 256, 256
local outputDir = "/home/saxiao/oir/data/res256/test/"

local function downSampling(raw, label, w, h, imageFileName)
  local draw = image.scale(raw, w, h)
  local labelDS = image.scale(label, w, h)
  local dlabel = labelDS.new():resizeAs(labelDS):zero()
  dlabel:maskedFill(labelDS:gt(0.5), 1)
  if imageFileName then
    local dlabeledImage = draw.new():resize(3, w, h):zero()
    dlabeledImage[1]:copy(draw)
    dlabeledImage[1]:maskedFill(dlabel, 255)
    dlabeledImage[2]:maskedFill(dlabel, 255)
    image.save(imageFileName, dlabeledImage)
  end

  return draw, dlabel
end

local function isGrey(image)
  local m12 = image[1]:eq(image[2])
  m12:maskedFill(m12:eq(0), 2)
  local m23 = image[2]:eq(image[3])
  local m123 = m12:eq(m23)  -- highlighted = 0, normal+background = 1
  local m1Not0 = image[1]:ne(0)
  local m123Not0 = m123:eq(m1Not0)
  local isGrey = false
  local labelMask = m123.new():resizeAs(m123):zero()
  labelMask:maskedFill(m123:eq(0), 1) -- hightlighted = 1, normal+bg = 0
  if m123Not0:sum() > 0 then
--    print(m123Not0:sum()/m12:size(1)/m12:size(2))
    isGrey =  true
    local diff = (image[1]:float():add(-1, image[2]:float())):abs()
    local redYellowCutoff = 30
    if diff:max() > redYellowCutoff then
      -- There are red and yellow highlighted
      labelMask:zero()
      labelMask:maskedFill(diff:lt(redYellowCutoff), 1)
      labelMask:maskedFill(m123, 0)  -- yellow =1, red+normal+bg=0
    end
  end
  return isGrey, labelMask
end

local function generateImage()
--  for i, rawPath in pairs(rawPathFile) do
  local N = 10 -- #rawPathFile
  for i = 1, N do
    local rawPath = rawPathFile[i]
    local rawImage = gm.Image(rawPath):toTensor('byte','RGB','DHW')
    local labelImage = gm.Image(labelPathFile[i]):toTensor('byte','RGB','DHW')
--    print(rawFile[indexes[i]])
--    print(labelFile[indexes[i]])
--    print(rawImage:size(), labelImage:size())
    if rawImage:size(2) == labelImage:size(2) then
      print(i)
      local label = rawImage[2]:ne(labelImage[2])  -- yellow = 1, the rest (normal, bg, red possibly) = 0
      local isGrey, qMask = isGrey(labelImage)
      if isGrey then
        label = qMask
      end

      local dataFileName = string.format("%s%d.t7", outputDir, i)
      local imageFileName = string.format("%splot/%d.png", outputDir, i)
      local draw, dlabel = downSampling(rawImage[1], label, imageW, imageH, imageFileName)
      local data = {}
      data.originalRawW = rawImage[1]:size(1)
      data.originalRawH = rawImage[1]:size(2)
      data.originalLabelW = label:size(1)
      data.originalLabelH = label:size(2)
      data.rawImageFile = rawPath
      data.rawLabelFile = labelPathFile[i]
      data.raw = draw
      data.label = dlabel
      torch.save(dataFileName, data)

    end
  end
  local stats = {}
  stats.nfiles = N 
  print(stats.nfiles)
  torch.save(string.format("%sstats.t7", outputDir), stats)
end

generateImage()
