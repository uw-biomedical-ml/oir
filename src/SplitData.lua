
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

local function splitData(raw, label, outputFile, indexes)
  local data = {}
  for i = 1, indexes:size(1) do
    data[i] = {raw = raw[indexes[i]], label = label[indexes[i]]}
  end
  torch.save(outputFile, data)
end

local dir = "/home/saxiao/oir/data/"
local rawFile = string.format("%sraw.txt", dir)
local labelFile = string.format("%slabel.txt", dir)
local raw, nfiles = loadFile(rawFile)
local label = loadFile(labelFile)
local indexes = torch.randperm(nfiles)
local trainSplit, validateSplit = 0.64, 0.16
local trainEndIdx = math.ceil(nfiles * trainSplit)
local validateEndIdx = math.ceil(nfiles * (trainSplit + validateSplit))
splitData(raw, label, string.format("%strain.t7", dir), indexes[{{1, trainEndIdx}}])
splitData(raw, label, string.format("%svalidate.t7", dir), indexes[{{trainEndIdx+1, validateIdx}}]
splitData(raw, label, string.format("%stest.t7", dir), indexes[{{validateIdx+1, nfiles}}] 

