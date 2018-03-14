require 'doPredict'
require 'lfs'

local cmd = torch.CmdLine()
cmd:option('--imageFolder', "sample/batch", 'image folder')
cmd:option('--task', 3, '1(yellow)|2(red)|3(both)')
cmd:option('--redModel', 'model/red.t7', 'trained model for red')
cmd:option('--yellowModel', 'model/yellow.t7', 'trained model for yellow')
cmd:option('--retinaModel', 'model/retina.t7')
cmd:option('--outputdir', 'output', 'output directory')
cmd:option('--nnRetina', 1, 'set 1 means to use the nn model to predict the whole retina')
cmd:option('--kmeansRetina', 0, 'set 1 means to use kmeans to predict the whole retina')
cmd:option('--gpu', -1, '-1 means using cpu, for i >= 0 means using gpu with id = i+1')
cmd:option('--thumbnailSize', 256, 'thumbnail size')
cmd:option('--uniqueName', 1, '1 means keeping the basename of the input image')
local opt = cmd:parse(arg)

local supportedFormat = {png=true, PNG=true, jpg=true, jpeg=true, JPG=true, JPEG=true, tif=true, tiff=true, TIF=true, TIFF=true}
local outputdirMap = {}
local originOutputdir = opt.outputdir
local function predictForDir(dir)
  local finalOutputdir = string.format("%s/result", dir)
  local outputdir = string.format("%s/%s/result", originOutputdir, dir)
  table.insert(outputdirMap, {tmp=outputdir, final=finalOutputdir})
  opt.outputdir = string.format("%s/image", outputdir)
  paths.mkdir(opt.outputdir)
  local html = io.open(paths.concat(outputdir, 'index.html'), 'w')
  html:write('<table style="text-align:center;">')
  html:write('<tr><td>file name</td><td>Input</td><td>Predict</td><td>VO pixels</td><td>NV pixels</td>')
  if opt.kmeansRetina > 0 then
    html:write('<td>Retina kmeans</td><td>Retina pixels</td><td>VO ratio</td><td>NV ratio</td>')
  end
  if opt.nnRetina > 0 then
    html:write('<td>Retina NN</td><td>Retina pixels</td><td>VO ratio</td><td>NV ratio</td>')
  end
  html:write('</tr>')
  
  local csvFile = io.open(string.format("%s/results.csv", outputdir), 'w')
  csvFile:write("filename,VO pixels,NV pixels")
  if opt.kmeansRetina > 0 then
    csvFile:write(",Retina(kmeans) pixels,VO ratio,NV ratio")
  end
  if opt.nnRetina > 0 then
    csvFile:write(",Retina(NN) pixels,VO ratio,NV ratio")
  end
  csvFile:write("\n")

  for file in lfs.dir(dir) do
    local suffix = string.match(file, ".*%.(.*)")
    if supportedFormat[suffix] then
      local filepath = string.format("%s/%s", dir, file)
      print(filepath)
      opt.imageFile = filepath
      local ratio, pixels = predict(opt)

      -- write to webpage
      local basename = paths.basename(file, suffix)
      html:write('<tr>')
      html:write('<td>' .. basename .. '</td>')
      html:write(string.format('<td><img src="./image/%s_thumbnail.png"/></td>', basename))
      html:write(string.format('<td><img src="./image/%s_quantified_thumbnail.png"/></td>', basename))
      html:write('<td>' .. pixels.VO .. '</td>')
      html:write('<td>' .. pixels.NV .. '</td>')
      csvFile:write(file)
      csvFile:write(string.format(",%d,%d", pixels.VO, pixels.NV))
      if opt.kmeansRetina > 0 then
        local voratio, nvratio = string.format("%.3f", ratio.VO.kmeans), string.format("%.3f", ratio.NV.kmeans)
        html:write(string.format('<td><img src="./image/%s_retina.png"/></td>', basename))
        html:write('<td>' .. pixels.retina.kmeans .. '</td>')
        html:write('<td>' .. voratio .. '</td>')
        html:write('<td>' .. nvratio .. '</td>')
        csvFile:write(string.format(",%d,%s,%s", pixels.retina.kmeans, voratio, nvratio))
      end
      if opt.nnRetina > 0 then
        local voratio, nvratio = string.format("%.3f", ratio.VO.nn), string.format("%.3f", ratio.NV.nn)
        html:write(string.format('<td><img src="./image/%s_retina.png"/></td>', basename))
        html:write('<td>' .. pixels.retina.nn .. '</td>')
        html:write('<td>' .. voratio .. '</td>')
        html:write('<td>' .. nvratio .. '</td>')
        csvFile:write(string.format(",%d,%s,%s", pixels.retina.nn, voratio, nvratio))
      end
      html:write('</tr>')
      csvFile:write("\n")
    end
  end
  html:write('</table>')
  html:close()
  csvFile:close()
end

function predictRecursive(dir)
  local lastlayer = true
  for file in lfs.dir(dir) do
    if file ~= "." and file ~= ".." then
      local filepath = dir .. "/" .. file
      if lfs.attributes(filepath, "mode") == "directory" then
        lastlayer = false
        predictRecursive(filepath)
      end
    end
  end
  
  if lastlayer then 
    predictForDir(dir) 
  end
end

predictRecursive(opt.imageFolder)
