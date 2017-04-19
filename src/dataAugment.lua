require 'torch'
require 'image'

local utils = require 'utils'

local N = 1
local mode = 'simple'
local dataDir = "/home/saxiao/oir/data/train/"
local outputDir = "/home/saxiao/oir/data/augment/train/"
local plotDir = "/home/saxiao/oir/data/augment/train/plot/"

local function rotateImage(raw, label, imageFileRoot)
  local theta = torch.uniform(0, 2 * math.pi)
  local rawRotated = image.rotate(raw, theta, mode)
  local labelRotated = image.rotate(label, theta, mode)
  if imageFileRoot then
    utils.drawImage(string.format("%s.png", imageFileRoot), raw, label)
    utils.drawImage(string.format("%s_%.3f.png", imageFileRoot, theta), rawRotated, labelRotated)
  end
  return rawRotated, labelRotated, theta
end

local function generate()
  local stats = torch.load(dataDir .. "/stats.t7")
  local nfiles = 1  -- stats.nfiles
  local cnt = 0
  for f = 1, nfiles do
    local fileData = torch.load(string.format("%s%d.t7", dataDir, f))
    local raw, label = fileData.raw, fileData.label
    print(raw:size())
    for i = 1, N do
      cnt = cnt + 1
      local imageFileRoot = string.format("%s%d", plotDir, f)
      local rawRotated, labelRotated, theta = rotateImage(raw, label, imageFileRoot)
      local data = {}
      data.raw = rawRotated
      data.label = labelRotated
      data.theta = theta
      data.rawImageFile = fileData.rawImageFile
      data.rawLabelFile = fileData.rawLabelFile
      torch.save(string.format("%s%d.t7", outputDir, cnt), data)
    end
  end
  stats = {}
  stats.nfiles = cnt
  print(stats.nfiles)
  torch.save(string.format("%sstats.t7", outputDir), stats)
end

generate()