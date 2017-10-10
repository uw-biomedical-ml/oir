require 'lfs'

local function loadIndex(txtFile)
  local t = {}
  local cnt = 0
  local file = io.open(txtFile)
  if file then
    for line in file:lines() do
      table.insert(t, tonumber(line))
      cnt = cnt + 1
    end
  end
  return t, cnt
end

local output = {}

local function addData(datafile, retinafile, indexes)
  local data = torch.load(datafile)
  local retina = torch.load(retinafile)
  for _, i in pairs(indexes) do
    o = {}
    o.input = data[i].input
    o.target = retina[i].retina
    o.centroidsLog = retina[i].centroidsLog
    o.labelFilePath = data[i].labelFilePath
    o.rawFilePath = data[i].rawFilePath
    o.originalSize = data[i].originalSize
    table.insert(output, o)
  end
end

local function createTrainset()
  local trainIndexes = loadIndex("output/retina/train/select.txt")
  addData("data/res256/train.t7", "output/retina/train/retina.t7", trainIndexes)

  local testIndexes = loadIndex("output/retina/test/select.txt")
  addData("data/res256/test.t7", "output/retina/test/retina.t7", testIndexes)

  torch.save("data/retina/train.t7", output)
end

local function createValidset()
  local valIndexes = loadIndex("output/retina/validate/select.txt")
  addData("data/res256/validate.t7", "output/retina/validate/retina.t7", valIndexes)
  torch.save("data/retina/validate.t7", output)
end

createTrainset()
--createValidset()
