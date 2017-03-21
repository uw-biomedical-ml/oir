require 'lfs'
require 'nn'
require 'nngraph'
local utils = require 'src/utils'

local useNN = true
local model, dtype
if useNN then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu+1)
  cutorch.manualSeed(123)
  dtype = 'torch.CudaTensor'
  
  model = torch.load("")
  model = model:type(dtype)
end

local function learnRetina(img2D, html, outpudir, basename)
  local retina_km = utils.learnByKmeansThreshold(img2D, {k=2, verbose=true})
  utils.drawImage(string.format("%s/image/%s_raw.png", outputdir, basename), img2D:byte())
  utils.drawImage(string.format("%s/image/%s_retina_km.png", outputdir, basename), img2D:byte(), retina_km)
  local originalType = img2D:type()
  if model then
    model:evaluate()
    img2D = img2D:type(dtype)
    retina_nn = model:forward(img2D:view(1,1,img2D:size(1),-1))
    utils.drawImage(string.format("%s/image/%s_retina_nn.png", outputdir, basename), img2D:type(originalType):byte(), retina_nn:type(originalType):byte())
  end
end

local function learnRetina(img2D, html, outputdir, basename)
  local retina = utils.learnByKmeansThreshold(img2D, {k=2, verbose=true})
  utils.drawImage(string.format("%s/image/%s_raw.png", outputdir, basename), img2D:byte())
  utils.drawImage(string.format("%s/image/%s_retina.png", outputdir, basename), img2D:byte(), retina)
  html:write('<tr>')
  html:write('<td>' .. basename .. '</td>')
  html:write(string.format('<td><img src="./image/%s_raw.png"/></td>', basename))
  html:write(string.format('<td><img src="./image/%s_retina.png"/></td>', basename))
  html:write('</tr>')
  return retina
end

local rootdir = 'output/retina'
local split = "test"
local outputdir = string.format("%s/%s", rootdir, split)
paths.mkdir(outputdir .. '/image')
local datafile = string.format("data/res256/%s.t7", split)
local data = torch.load(datafile)
local html = io.open(paths.concat(outputdir, 'index.html'), 'w')
html:write('<table style="text-align:center;">')
html:write('<tr><td>file name</td><td>Input</td><td>Predict Retina</td></tr>')
local output = {}
for i=1, #data do
  local retina = learnRetina(data[i].input, html, outputdir, i)
  table.insert(output, retina)
end
html:write('</table>')
html:close()
torch.save(string.format("%s/retina.t7", outputdir), output)
