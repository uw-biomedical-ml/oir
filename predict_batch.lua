require 'predict'
require 'lfs'

local opt = {task = 3, 
             redModel = 'model/red.t7',
             yellowModel = 'model/yellow.t7',
             learnRetina = true,
             verbose = true,
             gpu = 0,
             thumbnailSize = 256,
             nnRetina = true,
             kmeansRetina = false,
             retinaModel = 'model/retina.t7'}

local outputdirMap = {}
local function predictForDir(dir)
  local finalOutputdir = string.format("%s/result", dir)
  local outputdir = string.format("output/%s/result", dir)
  table.insert(outputdirMap, {tmp=outputdir, final=finalOutputdir})
  opt.outputdir = string.format("%s/image", outputdir)
  paths.mkdir(opt.outputdir)
  local html = io.open(paths.concat(outputdir, 'index.html'), 'w')
  html:write('<table style="text-align:center;">')
  html:write('<tr><td>file name</td><td>Input</td><td>Predict</td><td>VO pixels</td><td>NV pixels</td>')
  if opt.kmeansRetina then
    html:write('<td>Retina kmeans</td><td>Retina pixels</td><td>VO ratio</td><td>NV ratio</td>')
  end
  if opt.nnRetina then
    html:write('<td>Retina NN</td><td>Retina pixels</td><td>VO ratio</td><td>NV ratio</td>')
  end
  html:write('</tr>')
  
  local csvFile = io.open(string.format("%s/results.csv", outputdir), 'w')
  csvFile:write("filename,VO pixels,NV pixels")
  if opt.kmeansRetina then
    csvFile:write(",Retina(kmeans) pixels,VO ratio,NV ratio")
  end
  if opt.nnRetina then
    csvFile:write(",Retina(NN) pixels,VO ratio,NV ratio")
  end
  csvFile:write("\n")

  for file in lfs.dir(dir) do
    print(file)
    if string.match(file, 'tif') then
      local filepath = string.format("%s/%s", dir, file)
      print(filepath)
      opt.imageFile = filepath
      local ratio, pixels = predict(opt)

      --os.execute("cp \"" .. filepath .. "\" \"" .. opt.outputdir .. "\"")  

      -- write to webpage
      local basename = paths.basename(file, 'tif')
      html:write('<tr>')
      html:write('<td>' .. basename .. '</td>')
      html:write(string.format('<td><img src="./image/%s_thumbnail.png"/></td>', basename))
      html:write(string.format('<td><img src="./image/%s_quantified_thumbnail.png"/></td>', basename))
      html:write('<td>' .. pixels.VO .. '</td>')
      html:write('<td>' .. pixels.NV .. '</td>')
      csvFile:write(file)
      csvFile:write(string.format(",%d,%d", pixels.VO, pixels.NV))
      if opt.kmeansRetina then
        local voratio, nvratio = string.format("%.3f", ratio.VO.kmeans), string.format("%.3f", ratio.NV.kmeans)
        html:write(string.format('<td><img src="./image/%s_retina_kmeans.png"/></td>', basename))
        html:write('<td>' .. pixels.retina.kmeans .. '</td>')
        html:write('<td>' .. voratio .. '</td>')
        html:write('<td>' .. nvratio .. '</td>')
        csvFile:write(string.format(",%d,%s,%s", pixels.retina.kmeans, voratio, nvratio))
      end
      if opt.nnRetina then
        local voratio, nvratio = string.format("%.3f", ratio.NV.nn), string.format("%.3f", ratio.NV.nn)
        html:write(string.format('<td><img src="./image/%s_retina_nn.png"/></td>', basename))
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
    print(dir)
    predictForDir(dir) 
  end
end

local dir = "/data/oir/data/OIR quantification for Felicitas/IfnG P17 inj P14/20170206" --"/data/oir/data/OIR Quantification for Kyle"
predictRecursive(dir)
torch.save("outputdirMapping.t7", outputdirMap)
