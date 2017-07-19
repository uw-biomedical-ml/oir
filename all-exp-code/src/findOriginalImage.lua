require 'torch'
require 'os'

local indexFile = torch.load("/home/saxiao/oir/src/originalIndex_test.t7")
local start, N = 1, 3 -- indexFile:size(1)

for i = start, N do
  local fileName = string.format("/home/saxiao/oir/data/res256/test/%d.t7", indexFile[i])
  local f = torch.load(fileName)
  local from = f.rawImageFile
  print(f.originalRawW, from)
  
  local to = string.format("/home/saxiao/original/epoch_60_%d_or.tif", i)
  os.execute("cp \"" .. from .. "\" \"" .. to .. "\"")
  from = f.rawLabelFile
  to = string.format("/home/saxiao/original/epoch_60_%d_ot.tif", i) 
  os.execute("cp \"" .. from .. "\" \"" .. to .. "\"")
end
