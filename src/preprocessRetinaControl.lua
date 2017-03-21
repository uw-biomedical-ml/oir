require 'lfs'
local gm = require 'graphicsmagick'
local utils = require 'src/utils'

local dataDir = "/data/oir/retina-segmented"

local function findMatchedRawFile(files, fileName, key)
  if string.match(fileName, key) then
    local qStart = string.find(fileName, key)
    local rawFilePrefix = string.sub(fileName,1,qStart-1)
    rawFilePrefix = string.gsub(rawFilePrefix, "%s*$", "")
    local tmpFile = "tmp.txt"
    os.execute("find /data/oir/ -name '*" .. rawFilePrefix .. "*' > " .. tmpFile)
    local file = io.open(tmpFile)
    for line in file:lines() do
      if not string.match(line, key) then
        table.insert(files, {raw=line, label=string.format("%s/%s", dataDir, fileName)})
        break
      end
    end
  end
end

local files = {}
for file in lfs.dir(dataDir) do
  utils.findMatchedRawFile(files, file, "quantified")
  utils.findMatchedRawFile(files, file, "labeled")
end

local function isRetinaBoundary(rgb)
  return rgb[3] > 200
end

--torch.save("/home/saxiao/oir/data/retinaPath.t7", files)
for i = 10, #files do
  print(i)
  local path = files[i]
  local rawImg = gm.Image(path.raw):toTensor('byte','RGB','DHW')  
  local labelImg = gm.Image(path.label):toTensor('byte', 'RGB', 'DHW')
  local retinaLabel = utils.fillDfs(labelImg, isRetinaBoundary)
  local fileName = string.format("/home/saxiao/oir/data/retina/%d.t7", i)
  torch.save(fileName, {input=rawImg[1], target=retinaLabel})
end
