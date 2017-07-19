
local dataDir = "/home/saxiao/oir/data/res256/"
local imageW, imageH = 256, 256
local mode = "test"
local trainSplit = 0.8

local statFileName = string.format("%s%s/stats.t7", dataDir, mode)
local statFile = torch.load(statFileName)
local N = statFile.nfiles
local indexes = torch.randperm(N)

local function merge(indexArray, split)
local n = indexArray:size(1)
local raw = torch.ByteTensor(n, 1, imageW, imageH)
local label = torch.ByteTensor(n, imageW, imageH)
local originalRawFile, originalLabelFile, originalRawW, originalRawH, originalLabelW, originalLabelH = {}, {}, {}, {}, {}, {}
for i = 1, n do
  local dataFileName = string.format("%s%s/%d.t7", dataDir, mode, indexArray[i])
  print(dataFileName)
  local dataFile = torch.load(dataFileName)
  raw[i][1] = dataFile.raw
  label[i] = dataFile.label
  originalRawFile[i] = dataFile.rawImageFile
  originalLabelFile[i] = dataFile.rawLabelFile
end

local toSave = {}
toSave.raw = raw
toSave.label = label
toSave.originalRawFile = originalRawFile
toSave.originalLabelFile = originalLabelFile
toSave.originalRawW = originalRawW
toSave.originalRawH = originalRawH
toSave.originalLabelW = originalLabelW
toSave.originalLabelH = originalLabelH
torch.save(string.format("%s%s.t7", dataDir, split), toSave)
end

--local trainN = math.ceil(N * trainSplit)
--merge(indexes[{{1, trainN}}], "train")
--print("validate")
--merge(indexes[{{trainN+1, N}}], "validate")
merge(indexes, "test")
