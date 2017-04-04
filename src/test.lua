require 'torch'
require 'gnuplot'
require 'image'

local gm = require 'graphicsmagick'

local Loader = require 'Loader'
local dir = "/home/saxiao/eclipse/workspace/oir/"

local function drawImage(raw, label, fileName)
  local labeledImage = raw.new():resize(3, raw:size(1), raw:size(2)):zero()
  labeledImage[1]:copy(raw)
  labeledImage[1]:maskedFill(label, 255)
  labeledImage[2]:maskedFill(label, 255)
  image.save(fileName, labeledImage)
end

local function downSampling(raw, label, dataFileName, w, h, imageFileName)
  local draw = image.scale(raw, w, h)
  local labelDS = image.scale(label, w, h)
  local dlabel = labelDS.new():resizeAs(labelDS):zero()
  dlabel:maskedFill(labelDS:gt(0.5), 1)
  local data = {}
  data.raw = draw
  data.label = dlabel
  torch.save(dataFileName, data)
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
    print(m123Not0:sum()/m12:size(1)/m12:size(2))
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

local function inspectImages()
  local opt = {}
  opt.dataDir = "/home/saxiao/oir/CNTF"
  opt.pathsFile = dir .. "data/cntf.txt"
  opt.batchSize = 1
  opt.trainSize = 1
  opt.validateSize = 0
  local loader = Loader.create(opt)
  
  local nImages = #loader.allData["train"]
  local w, h = 256, 256
  local start, endIndex = 1, nImages
  for i = start, endIndex do
    local qfile = loader.allData["train"][i]
    print(string.format("i=%d",i), qfile)
    local qStart = string.find(qfile, "quantified")
    local rawfile = string.sub(qfile,1,qStart-1)
    rawfile = string.gsub(rawfile, "%s*$", "") .. ".tif"
    local raw = gm.Image(rawfile):toTensor('byte','RGB','DHW')
    local labeled = gm.Image(qfile):toTensor('byte','RGB','DHW')
    local label = raw[2]:ne(labeled[2])  -- yellow = 1, the rest (normal, bg, red possibly) = 0
    local isGrey, qMask = isGrey(labeled)
    if isGrey then
      label = qMask
    end
    
--    local fileName = dir .. "plot/cntf/" .. i .. ".png"
--    drawImage(raw[1], label, fileName)
    local dataFileName = dir .. "data/cntf/" .. i .. ".t7"
    downSampling(raw[1], label, dataFileName, w, h)
  end
  print(nImages)
  
end

--inspectImages()

--local stats = {}
--stats.nfiles = 195
--torch.save(dir .. "data/cntf/stats.t7", stats)

--local dir = "/home/saxiao/eclipse/workspace/oir/"
--local file = torch.load(dir .. "data/allPaths.t7")
--local qfile = file[1]
--local qStart = string.find(qfile, "quantified")
--local rawfile = string.sub(qfile,1,qStart-1)
--rawfile = string.gsub(rawfile, "%s$", "") .. ".tif"
--print(rawfile) 
--
--local raw = gm.Image(rawfile):toTensor('byte','RGB','DHW')
--local q = gm.Image(qfile):toTensor('byte','RGB','DHW')
--local labels = torch.eq(raw[2], q[2])

--local rawImage = raw.new():resizeAs(raw):zero()
--rawImage[1]:copy(raw[1])
--local labeledImage = raw.new():resizeAs(raw):zero()
--labeledImage[1]:copy(raw[1])
--labeledImage[1]:maskedFill(labels:eq(0), 255)
--labeledImage[2]:maskedFill(labels:eq(0), 255)
--local rawFile = dir .. "plot/raw.png"
--image.save(rawFile, rawImage)
--local trueFile = dir .. "plot/true.png"
--image.save(trueFile, labeledImage)

--local w, h = 256, 256
--local downsampledImage = raw.new():resize(3, w, h):zero()
--downsampledImage[1]:copy(image.scale(raw[1], w, h))
--local downsampledFile = dir .. "plot/downsampled.png"
--image.save(downsampledFile, downsampledImage)

--local w = raw:size(2)
--local patchSize = 64
--local n = 1180
--local centerX = (patchSize * n - patchSize/2) % w 
--local centerY = patchSize * ((patchSize * n - patchSize/2) / w + 1) - patchSize/2
--print(centerX, centerY)
--
--local rsub1 = raw[1][{{centerX-patchSize/2+1,centerX+patchSize/2},{centerY-patchSize/2+1,centerY+patchSize/2}}]
--local qsub1 = labels[{{centerX-patchSize/2+1,centerX+patchSize/2},{centerY-patchSize/2+1,centerY+patchSize/2}}]
--
--local savefile = dir .. "plot/raw" .. centerX .. "_" .. centerY .. ".png"
--gnuplot.pngfigure(savefile)
--gnuplot.imagesc(rsub1)
--gnuplot.plotflush()
--
--savefile = dir .. "plot/q" .. centerX .. "_" .. centerY .. ".png"
--gnuplot.pngfigure(savefile)
--gnuplot.imagesc(qsub1)
--gnuplot.plotflush()

