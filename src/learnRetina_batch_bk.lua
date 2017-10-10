require 'lfs'
local utils = require 'src/utils'

local function learnRetina(img2D, html, outputdir, basename)
  local retina, centroidsLog = utils.learnByKmeansThreshold(img2D, {k=2, verbose=true})
  utils.drawImage(string.format("%s/image/%s_raw.png", outputdir, basename), img2D:byte())
  utils.drawImage(string.format("%s/image/%s_retina.png", outputdir, basename), img2D:byte(), retina)
  html:write('<tr>')
  html:write('<td>' .. basename .. '</td>')
  html:write(string.format('<td><img src="./image/%s_raw.png"/></td>', basename))
  html:write(string.format('<td><img src="./image/%s_retina.png"/></td>', basename))
  html:write('</tr>')
  return retina, centroidsLog
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
  local retina, centroidsLog = learnRetina(data[i].input, html, outputdir, i)
  table.insert(output, {retina=retina, centroidsLog=centroidsLog})
end
html:write('</table>')
html:close()
torch.save(string.format("%s/retina.t7", outputdir), output)
