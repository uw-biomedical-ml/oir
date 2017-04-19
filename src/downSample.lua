require 'torch'
require 'image'

local gm = require 'graphicsmagick'

local rootDir = "/home/saxiao/oir/"
local labelFileName = string.format("%sdata/labeled.txt", rootDir)
local rawFileName = string.format("%sdata/raw.txt", rootDir)

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

local labelFile, nLabel = loadFile(labelFileName)
local rawFile, nRaw = loadFile(rawFileName)
print(nLabel)
local trainSetPartition = 0.8
local nTrainSet = math.ceil(nLabel * trainSetPartition)
local indexes = torch.randperm(nLabel)
local imageW, imageH = 512, 512

local function generateImage(iStart, iEnd, outputDir)
  local rawFilePath, labelFilePath = {}, {}
  local cnt = 0
  for i = iStart, iEnd do
    local rawImage = gm.Image(rawFile[indexes[i]]):toTensor('byte','RGB','DHW')
    local labelImage = gm.Image(labelFile[indexes[i]]):toTensor('byte','RGB','DHW')
--    print(rawFile[indexes[i]])
--    print(labelFile[indexes[i]])
--    print(rawImage:size(), labelImage:size())
    if rawImage:size(2) == labelImage:size(2) then
      print(cnt)
      local label = rawImage[2]:ne(labelImage[2])  -- yellow = 1, the rest (normal, bg, red possibly) = 0
      local isGrey, qMask = isGrey(labelImage)
      if isGrey then
        label = qMask
      end
      
      table.insert(rawFilePath, rawFile[indexes[i]])
      table.insert(labelFilePath, labelFile[indexes[i]])
      cnt = cnt + 1
      local dataFileName = string.format("%s%d.t7", outputDir, cnt)
      local imageFileName = string.format("%splot/%d.png", outputDir, cnt)
      local draw, dlabel = downSampling(rawImage[1], label, imageW, imageH)
      local data = {}
      data.originalRawW = rawImage[1]:size(1)
      data.originalRawH = rawImage[1]:size(2)
      data.originalLabelW = label:size(1)
      data.originalLabelH = label:size(2)
      data.rawImageFile = rawFile[indexes[i]]
      data.rawLabelFile = labelFile[indexes[i]]
      data.raw = draw
      data.label = dlabel
      torch.save(dataFileName, data)

    end
  end
  local stats = {}
  stats.nfiles = cnt
  print(stats.nfiles)
  torch.save(string.format("%sstats.t7", outputDir), stats)
  torch.save(string.format("%srawPath.t7", outputDir), rawFilePath)
  torch.save(string.format("%slabelPath.t7", outputDir), labelFilePath)
end

generateImage(1, nTrainSet, string.format("%sdata/train/", rootDir))
generateImage(nTrainSet+1, nLabel, string.format("%sdata/test/", rootDir))
