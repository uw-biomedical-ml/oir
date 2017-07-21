require 'lfs'
require 'gnuplot'

local dirRoot = "/home/saxiao/oir/plot/yellow/"
local dataDir = dirRoot .. "test/sorted/"
local nbins = 40
local isUpsampled = true

local function findVal(file, key, istart, iend)
  local i, j = string.find(file, key)
  local val = string.sub(file, j+istart, j+iend)
  return val
end

local function findDiceVal(file, key)
  return findVal(file, key, 2, 6)
end

local function findRatioVal(file, key)
  return findVal(file, key, 14, 18)
end

local vals = {}
for file in lfs.dir(dataDir) do
  if lfs.attributes(dataDir .. file, "mode") == "file" and string.match(file, '_p_') then
    if isUpsampled then
      if string.match(file, 'upsampled') then
        table.insert(vals, findDiceVal(file, 'upsampled'))
        --table.insert(vals, findRatioVal(file, 'upsampled'))
      end
    else
      if not string.match(file, 'upsampled') then
        table.insert(vals, findDiceVal(file, '_p'))
        --table.insert(vals, findRatioVal(file, '_p'))
      end
    end
  end
end

print(#vals)
local valsVec = torch.Tensor(#vals)
for i, d in pairs(vals) do
  --print(d)
  valsVec[i] = d
end

print(valsVec:min(), valsVec:max())
local keyword = "dice"
local fileName = dirRoot .. keyword .. "HistTestSet.png"
if isUpsampled then fileName = dirRoot .. keyword .. "HistTestSet_upsampled.png" end
gnuplot.pngfigure(fileName)
gnuplot.title(isUpsampled and "dice histogram for the test set (upsampled)" or "dice histogram for the test set")
gnuplot.hist(valsVec, nbins)
--gnuplot.title(isUpsampled and "Predicted area/True area histogram for the test set (upsampled)" or "Predicted area/True area histogram for the test set")
gnuplot.hist(valsVec, nbins, 0, 1)
gnuplot.plotflush()

