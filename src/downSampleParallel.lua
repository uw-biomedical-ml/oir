require 'torch'
require 'image'
local tnt = require 'torchnet'
--local gm = require 'graphicsmagick'
local utils = require 'utils'

local rootDir = "/home/saxiao/oir/data/"
local imageW, imageH = 512, 512
local nThread = 3
local doDownsample = true
local outputDir = "/home/saxiao/oir/data/"
local saveSingleFile = false

local function downSampling(raw, label, w, h, plotOpt)
  local draw = image.scale(raw, w, h)
  local dlabel = draw.new():resizeAs(draw):zero()
  for i = 1, label:max() do
    local imask = image.scale(label:eq(i), w, h)
    dlabel:maskedFill(imask:gt(0.5), i)
  end
  if plotOpt then
    local imageFileRoot = string.format("%splot/%d", rootDir, plotOpt.idx)
    utils.drawImage(string.format("%s_t.png", imageFileRoot), draw, dlabel)
    local sortedOriginalLabel = string.format("%s_t_original.png", imageFileRoot)
    os.execute("cp \"" .. plotOpt.originalLabelFile .. "\" \"" .. sortedOriginalLabel .. "\"")
    utils.drawImage(string.format("%s_r.png", imageFileRoot), draw)
    local sortedOriginalRaw = string.format("%s_r_original.png", imageFileRoot)
    os.execute("cp \"" .. plotOpt.originalRawFile .. "\" \"" .. sortedOriginalRaw .. "\"")
  end

  return draw, dlabel
end

local function getDownsampleIter(inputFile)
  local pathFile = torch.load(inputFile)
  local nfiles = #pathFile
  return tnt.ParallelDatasetIterator{
    nthread = nThread,
    init = function()
      require 'torchnet'
      require 'image'
      gm = require 'graphicsmagick'
    end,
    ordered = true,
    closure = function()
      return tnt.ListDataset{
          list = torch.range(1, nfiles):long(),
          load = function(idx)
            local rawPath, labelPath = pathFile[idx].raw, pathFile[idx].label
            local rawImage = gm.Image(rawPath):toTensor('byte','RGB','DHW')
            local labelImage = gm.Image(labelPath):toTensor('byte','RGB','DHW')
            if rawImage:size(2) == labelImage:size(2) then
              local label = utils.getLabel(rawImage, labelImage) -- bg/normal = 1, yellow = 1, red = 2
              local hasRed = false
              if label:max() == 2 then
                hasRed = true
              end
              local input, target = rawImage[1], label
              if doDownsample then
                local plotOpt = {idx=idx, originalLabelFile=labelPath, originalRawFile=rawPath}
                input, target = downSampling(rawImage[1], label, imageW, imageH)
              end
              return {input = input, target = label, rawFilePath = rawPath, labelFilePath = labelPath, originalSize = {w = label:size(1), h = label:size(2)}, hasRed = hasRed}
            else
              return nil
            end
          end
      }
    end,
  }  
end

local function generate(split)
  print(split)
  local inputFile = string.format("%s%s_path.t7", rootDir, split)
  local pathFile = torch.load(inputFile)
  local iter = getDownsampleIter(inputFile)
  local data = {}
  local cnt, redCnt, i = 0, 0, 0
  for sample in iter() do
    i = i + 1
    assert(pathFile[i].raw == sample.rawFilePath, string.format("parallel iterator doesn't preserve the order, the %dth input was %s, output was %s", i, pathFile[i].raw, sample.rawFilePath))
    if sample then
      cnt = cnt + 1
      if sample.hasRed then 
        redCnt = redCnt + 1
        print(redCnt)
        if saveSingleFile then
          local fileName = string.format("%s/%s/%d.t7", outputDir, split, redCnt)
          torch.save(fileName, sample)
        else
          table.insert(data, sample)
        end
      end
    end
  end
  print(string.format("%s: %d %d", split, cnt, redCnt))
  if not saveSingleFile then
    print(string.format("output size: %d", #data))
    local outputFile = string.format("%sres%d/%s.t7", rootDir, imageW, split)
    torch.save(outputFile, data)
  end
end

generate("train")
generate("validate")
generate("test")

