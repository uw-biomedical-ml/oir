require 'predict'
require 'lfs'

local opt = {task = 3, 
             redModel = 'model/red.t7',
             yellowModel = 'model/yellow.t7',
             learnRetina = true,
             verbose = true,
             gpu = 0,
             thumbnailSize = 256,
             ks={2}}

local outputdirMap = {}
local function predictForDir(dir)
  local finalOutputdir = string.format("%s/result", dir)
  local outputdir = string.format("output/%s/result", dir)
  table.insert(outputdirMap, {tmp=outputdir, final=finalOutputdir})
  opt.outputdir = string.format("%s/image", outputdir)
  paths.mkdir(opt.outputdir)
  local html = io.open(paths.concat(outputdir, 'index.html'), 'w')
  html:write('<table style="text-align:center;">')
  html:write('<tr><td>file name</td><td>Input</td><td>Predict</td>')
  for _, k in pairs(opt.ks) do
    html:write(string.format("<td>Retina k=%d</td><td>VO ratio k=%d</td><td>NV ratio k=%d</td>", k,k,k))
  end
  html:write('</tr>')
  
  local csvFile = io.open(string.format("%s/results.csv", outputdir), 'w')
  csvFile:write("filename")
  for _, k in pairs(opt.ks) do
    csvFile:write(string.format(",voratio_k%d,nvratio_2%d", k, k))
  end
  csvFile:write("\n")

  for file in lfs.dir(dir) do
    print(file)
    if string.match(file, 'tif') then
      local filepath = string.format("%s/%s", dir, file)
      print(filepath)
      opt.imageFile = filepath
      local ratio = predict(opt)

      --os.execute("cp \"" .. filepath .. "\" \"" .. opt.outputdir .. "\"")  

      -- write to webpage
      local basename = paths.basename(file, 'tif')
      html:write('<tr>')
      html:write('<td>' .. basename .. '</td>')
      html:write(string.format('<td><img src="./image/%s_thumbnail.png"/></td>', basename))
      html:write(string.format('<td><img src="./image/%s_quantified_thumbnail.png"/></td>', basename))
      csvFile:write(file)
      for _, k in pairs(opt.ks) do
        local voratio, nvratio = string.format("%.3f", ratio[1][k]), string.format("%.3f", ratio[2][k])
        html:write(string.format('<td><img src="./image/%s_retina_k%d.png"/></td>', basename, k))
        html:write('<td>' .. voratio .. '</td>')
        html:write('<td>' .. nvratio .. '</td>')
        csvFile:write(string.format(",%s,%s", voratio, nvratio))
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

local dir = "/data/oir-test/OIR Quantification for Kyle"
predictRecursive(dir)
torch.save("outputdirMapping.t7", outputdirMap)
