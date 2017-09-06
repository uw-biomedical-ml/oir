require 'nn'
require 'nngraph'
require 'image'
require 'lfs'
local utils = require 'src/utils'
local gm = require 'graphicsmagick'
require 'cunn'
require 'cutorch'
  
cutorch.setDevice(1)
cutorch.manualSeed(123)
--local dtype = 'torch.DoubleTensor'
local  dtype = 'torch.CudaTensor'
local drawImage = true

local dataDir = "/data/oir-interperson/Interperson variability"

local fileCnt = 0
local files = {}
local fileName = "/home/saxiao/oir/data/interperson-files.t7"
local diceFile = 'interpersonDice.t7'

local function findLabels(filePath, key, labelId)
  for file in lfs.dir(filePath) do
    if file ~= "." and file ~= ".." then
      local path = filePath .. "/" .. file
      if lfs.attributes(path, "mode") == "directory" then
        findLabels(path, key, labelId)
      elseif lfs.attributes(path, "mode") == "file" and string.match(file, key) then
        local qStart = string.find(file, key)
        local rawFilePrefix = string.sub(file,1,qStart-1)
        rawFilePrefix = string.gsub(rawFilePrefix, "%s*$", "")
        local values = files[rawFilePrefix] 
        if not values then
          values = {}
          files[rawFilePrefix] = values
          fileCnt = fileCnt + 1
        end
        values[labelId] = path
      end
    end
  end
end

local rawCnt = 0
local function findMatchedRawFileOld(rawFilePrefix)
  local tmpFile = "tmp.txt"
  os.execute("find '/data/oir-interperson/Interperson variability/Sophia' -name '*" .. rawFilePrefix .. ".tif' > " .. tmpFile)
  local file = io.open(tmpFile)
  for line in file:lines() do
    rawCnt = rawCnt + 1
    return line
  end
end

local function findMatchedRawFile(rawFilePrefix)
  local tmpFile = "tmp.txt"
  os.execute("find '/data/oir-interperson/Originals' -name '" .. rawFilePrefix .. ".tif' > " .. tmpFile)
  local file = io.open(tmpFile)
  for line in file:lines() do
    rawCnt = rawCnt + 1
    return line
  end
end

local function createFileList(fileName)
  local dir = string.format("%s/Feli", dataDir)
  findLabels(dir, 'quantified', 'feli')
  dir = string.format("%s/Kyle/OIR miR34a mimic -InjP14 OIR P17 20160315", dataDir)
  findLabels(dir, '_quant', 'kyle')
  dir = string.format("%s/Kyle/P17 inj P12 miR34a mimic 20160608", dataDir)
  findLabels(dir, '.tif', 'kyle')
  dir = string.format("%s/Kyle/P17 inj P12 miR34a mimic 20160707", dataDir)
  findLabels(dir, '.tif', 'kyle')
  dir = string.format("%s/RJ", dataDir)
  findLabels(dir, 'Quant.jpg', 'rj')
  dir = string.format("%s/Sophia", dataDir)
  findLabels(dir, 'Yellow%-Red', 'sophia')
  findLabels(dir, 'Yellowred', 'sophia')
  findLabels(dir, 'Yellow redone%-Red', 'sophia')
  findLabels(dir, 'Yellow Red', 'sophia')
  findLabels(dir, 'yellow%-Red', 'sophia')
  findLabels(dir, 'Yellow%-red', 'sophia')

  for rawFilePrefix, values in pairs(files) do
    values.raw = findMatchedRawFile(rawFilePrefix)
  end

  torch.save(fileName, files)
end

local function loadImage(imageFile)
  local img = nil
  local suffix = string.match(imageFile, ".*%.(.*)")
  if suffix == 'tif' or suffix == 'tiff' then
    img = gm.Image(imageFile):toTensor('byte','RGB','DHW')
  else
    img = image.load(imageFile) * 255
    img = img:byte()
  end
  return img
end

local function inference(img2D, model, dsize, targetLabel)
  model:evaluate()

  local dimg = image.scale(img2D, dsize, dsize)
  local originalType = dimg:type()
  dimg = dimg:type(dtype)
  local output = model:forward(dimg:view(1,1,dsize,-1))
  local upPredict = utils.upsampleLabel(output:view(dsize,dsize,-1), {h=img2D:size(1), w=img2D:size(2)}) - 1
  upPredict:maskedFill(upPredict:eq(1), targetLabel)
  return upPredict:byte()
end

local yellowModel = torch.load("model/yellow.t7"):type(dtype)
local redModel = torch.load("model/red.t7"):type(dtype)

local persons = {'feli', 'kyle', 'rj', 'sophia'}
local processedFile = "interperson.t7"
local cnt = 0
local function process()
  local files = torch.load(fileName)
  local rlt = {}
  for key, values in pairs(files) do
    if values.raw then
      cnt = cnt + 1
      print(values)
      local rawImage = loadImage(values.raw)
      local labels = {}
      labels.predictyellow = inference(rawImage[1], yellowModel, 256, 1)
      labels.predictred = inference(rawImage[1], redModel, 512, 2)
      if drawImage then
        utils.drawImage(string.format("plot/interperson/%d-yellow.png", cnt), rawImage[1], labels.predictyellow)
        utils.drawImage(string.format("plot/interperson/%d-red.png", cnt), rawImage[1], labels.predictred)
      end
      for _, person in pairs(persons) do
        local labelImage = loadImage(values[person])
        if labelImage:size(2) ~= rawImage:size(2) then
          labelImage = image.scale(labelImage, rawImage:size(2), rawImage:size(3))
        end
        local label = utils.getLabel(rawImage, labelImage, true) -- bg/normal = 0, yellow = 1, red = 2
        labels[person] = label
        if drawImage then
          utils.drawImage(string.format("plot/interperson/%d-%s.png", cnt, person), rawImage[1], label)
        end
        local size = rawImage:size(2)
      end
      rlt[key] = labels
      --if cnt > 0 then break end
    end
  end 
  torch.save(processedFile, rlt)
end

local function diceCoef(predict, target, targetLabel)
  local p = predict:eq(targetLabel):float()
  local t = target:eq(targetLabel):float()
  local pt = torch.cmul(p,t)
  local eps = 1
  return (pt:sum()*2 + eps) / (p:sum() + t:sum() + eps)
end

local function calDice(f, goldtruth, tocompare, targetLabel)
  print(goldtruth, tocompare)
  local dice = {}
  for key, values in pairs(f) do
    print(key)
    local predict = values[tocompare]
    local truth = values[goldtruth]
    local dc = diceCoef(predict, truth, targetLabel)
    table.insert(dice, dc)
  end
  
  dice = torch.Tensor(dice)
  return dice
end

local predictKey = {'predictyellow', 'predictred'}
local function diceForTarget(f, goldtruth, targetLabel)
  local rlt = {}
  rlt.model = calDice(f, goldtruth, predictKey[targetLabel], targetLabel)
  for _, person in pairs(persons) do
    if person ~= goldtruth then
      rlt[person] = calDice(f, goldtruth, person, targetLabel)
    end
  end
  return rlt
end

local function computeDice()
  local f = torch.load(processedFile)
  local dice = {}
  local cnt = 0
  for _, person in pairs(persons) do
    local yellow = diceForTarget(f, person, 1)
    local red = diceForTarget(f, person, 2)
    dice[person] = {yellow=yellow, red=red}
    cnt = cnt + 1
    --if cnt > 0 then break end
  end
  torch.save(diceFile, dice)
end

local function plotDiceHist()
  local f = torch.load(diceFile)
  for gt, values in pairs(f) do
    for target, dices in pairs(values) do
      for person, vals in pairs(dices) do
        local plotName = string.format("plot/interperson/dice/%s_%s_%s_dc.png", target, gt, person)
        gnuplot.pngfigure(plotName)
        gnuplot.title(string.format("%s - %s, %s", gt, person, target))
        gnuplot.hist(vals, 20, 0, 1)
        gnuplot.plotflush()
      end
    end
  end
end

local function plotDiceCompare()
  local f = torch.load(diceFile)
  for gt, values in pairs(f) do
    for target, dices in pairs(values) do
      local means, stds = {}, {}
      local xtics = {}
      -- make the model as the first entry
      table.insert(means, dices.model:mean())
      table.insert(stds, dices.model:std())
      table.insert(xtics, "'model' 1")
      local cnt = 1
      for person, vals in pairs(dices) do
        if person ~= 'model' then
          cnt = cnt + 1
          table.insert(means, vals:mean())
          table.insert(stds, vals:std())
          table.insert(xtics, string.format("'%s' %d", person, cnt))
        end
      end
      local v = torch.Tensor(cnt, 2)
      v[{{},1}] = torch.Tensor(means)
      v[{{},2}] = torch.Tensor(stds)
      local plotName = string.format("plot/interperson/dice/%s_%s_dice_compare.png", target, gt)
      gnuplot.pngfigure(plotName)
      gnuplot.title(string.format("gold truth: %s, %s", gt, target))
      gnuplot.plot(torch.range(1,cnt), v, 'w yerr')
      gnuplot.raw(string.format("set xtics (%s)", table.concat(xtics, ',')))
      gnuplot.axis{0,5,0,1}
      gnuplot.plotflush() 
    end
  end
end

local function plotLabeledImage()
  local f = torch.load(processedFile)
  local pathfile = torch.load(fileName)
  local dice = torch.load(diceFile)
  local cnt = 0
  for key, labels in pairs(f) do
    cnt = cnt + 1
    print(cnt)
    local rawImage = loadImage(pathfile[key].raw)
    local plotname = string.format("plot/interperson/%d_model_yellow_%.3f_%.3f_%.3f_.3f.png", cnt, dice.feli.yellow.model[cnt], dice.kyle.yellow.model[cnt], dice.rj.yellow.model[cnt], dice.sophia.yellow.model[cnt])
    utils.drawImage(plotname, rawImage[1], labels.predictyellow)
    plotname = string.format("plot/interperson/%d_model_red_%.3f_%.3f_%.3f_%.3f.png", cnt, dice.feli.red.model[cnt], dice.kyle.red.model[cnt], dice.rj.red.model[cnt], dice.sophia.red.model[cnt])
    utils.drawImage(plotname, rawImage[1], labels.predictred)
    for _, person in pairs(persons) do
      plotname = string.format("plot/interperson/%d_%s.png", cnt, person)
      utils.drawImage(plotname, rawImage[1], labels[person])
    end
  end
end

local function saveToFile(filename, segment, branch)
  local f = torch.load(filename)
  for gt, data in pairs(f) do
    local fname = string.format("plot/interperson/dice/%s-%s.txt", gt, segment)
    local file = io.open(fname, 'w')
    if branch then data = data[branch] end
    for person, dices in pairs(data) do
      for i=1, dices:size(1) do
        file:write(string.format("%s %f\n", person, dices[i]))
      end
    end
    file:close()
  end
end

local function saveDiceToFile()
  saveToFile(diceFile, 'yellow', 'yellow')
  saveToFile(diceFile, 'red', 'red')
  --local f = torch.load(diceFile)
  --for gt, data in pairs(f) do
  --  local fname = string.format("plot/interperson/dice/%s-yellow.txt", gt)
  --  local file = io.open(fname, 'w')
  --  for person, dices in pairs(data.yellow) do
  --    for i=1, dices:size(1) do
  --      file:write(string.format("%s %f\n", person, dices[i]))
  --    end
  --  end
  --  file:close()
  --  fname = string.format("plot/interperson/dice/%s-red.txt", gt)
  --  file = io.open(fname, 'w')
  --  for person, dices in pairs(data.red) do
  --    for i=1, dices:size(1) do
  --      file:write(string.format("%s %f\n", person, dices[i]))
  --    end
  --  end
  --  file:close()
  --end
end

local function calMeanMedian(model)
  print(model)
  local f = torch.load(diceFile)
  local allperson, allmodel
  for gt, data in pairs(f) do
    for person, dice in pairs(data[model]) do
      if person ~= 'model' then
        if allperson then
          allperson = torch.cat(allperson, dice)
        else
          allperson = dice
        end
      else
        if allmodel then
          allmodel = torch.cat(allmodel, dice)
        else
          allmodel = dice
        end
      end
    end
  end
  print("person", allperson:mean(), allperson:median():squeeze())
  print("model", allmodel:mean(), allmodel:median():squeeze())
end

local retinaTensorName = "interperson/retinaTensor.t7"
local retinafilename = "interperson/retinaArea.t7"
paths.mkdir("plot/interperson/retina")
paths.mkdir("interperson")
local N = -1

local function isRetinaBoundary(rgb)
 -- return math.abs(rgb[3]-rgb[2])/rgb[3] < 0.3 and math.abs(rgb[3]-rgb[1])/rgb[3] > 0.6
  return rgb[1]<10 and rgb[2]>50 and rgb[3]>50
end
persons = {'feli', 'rj'} -- rj:512, feli: 1024,
local printRaw = true
local function learnRetina()
  local file = torch.load(fileName)
  local retinafile = {model={}, feli={}, kyle={}, rj={}, sophia={}}
  local retinaTensor = {model={}, feli={}, kyle={}, rj={}, sophia={}}
  local cnt = 1
  for key, values in pairs(file) do
    if key ~= "Mouse7R Scramble 10x 6t6" and key ~= "Mouse2L pro miR34a" then
      print(cnt, key)
      local raw = loadImage(values.raw)
      if printRaw then
        utils.drawImage(string.format("plot/interperson/retina/%d_raw.png", cnt), raw[1]:byte())
      end
      local draw = image.scale(raw[1], 256, 256)
      local retina = utils.learnByKmeansThreshold(draw)
      retina = image.scale(retina, raw:size(2), raw:size(3), 'simple')
      table.insert(retinaTensor.model, retina)
      table.insert(retinafile.model, retina:sum())
      --table.insert(retinafile.model, retina:sum()*raw:size(2)*raw:size(3)/retina:size(1)/retina:size(2))
      utils.drawImage(string.format("plot/interperson/retina/%d_p.png", cnt), raw[1]:byte(), retina)
      for _, person in pairs(persons) do
        print(person)
        local img = loadImage(values[person])
        local dimg
        if person == 'feli' then
          if cnt == 13 then
            dimg = img
          else
            dimg = image.scale(img, 1024, 1024)
          end
        else
          dimg = image.scale(img, 512, 512)
        end
        --image.save(string.format("plot/interperson/retina/%d_%s_%s.png", cnt, person, key), dimg)
        retina = utils.fillDfs(dimg, isRetinaBoundary)
        if img:size(2) ~= retina:size(1) then
          retina = image.scale(retina, img:size(2), img:size(3), 'simple')
        end
        table.insert(retinaTensor[person], retina)
        table.insert(retinafile[person], retina:sum())
        --table.insert(retinafile[person], retina:sum()*img:size(2)*img:size(3)/retina:size(1)/retina:size(2))
        utils.drawImage(string.format("plot/interperson/retina/%d_%s.png", cnt, person), img[1]:byte(), retina)
      end
    end
    if cnt == N then break end
    cnt = cnt + 1
  end
  --torch.save(retinaTensorName, retinaTensor)
  --torch.save(retinafilename, retinafile)  
end

N = -1
local function newRetinaSegment()
  local file = torch.load(fileName)
  local retinaTensor = torch.load(retinaTensorName)
  local retinafile = torch.load(retinafilename)
  local cnt = 1
  local persons = {'sophia'}
  for _, person in pairs(persons) do
    retinaTensor[person] = {}
    retinafile[person] = {}
  end
  local c = 1
  for key, values in pairs(file) do
    if key ~= "Mouse7R Scramble 10x 6t6" and key ~= "Mouse2L pro miR34a" then
      print(cnt)
      local targetsize = retinaTensor.model[c]:size(1)
      c = c + 1
      print("target size", targetsize)
      for _, person in pairs(persons) do
        local img = loadImage(values[person])
        print("img size", img:size(2))
        local retina = 1 - torch.cmul(torch.cmul(img[1]:eq(0), img[2]:eq(255)), img[3]:eq(0))
        if img:size(2) ~= targetsize then
          retina = image.scale(retina, targetsize, targetsize, 'simple')
          img = image.scale(img, targetsize, targetsize)
        end
        table.insert(retinaTensor[person], retina)
        table.insert(retinafile[person], retina:sum())
        utils.drawImage(string.format("plot/interperson/retina/%d_%s.png", cnt, person), img[1]:byte(), retina)
      end
    end
    if cnt == N then break end
    cnt = cnt + 1
  end 
  torch.save(retinaTensorName, retinaTensor)
  torch.save(retinafilename, retinafile)
end

local segmentareafile = "interperson/segmentArea.t7"
persons = {'feli','kyle','rj', 'sophia'}
local function computeSegmentationArea()
  local f = torch.load(fileName)
  local file = torch.load(processedFile)
  local area = {yellow={}, red={}}
  area.yellow={model={}, feli={}, kyle={}, rj={}, sophia={}}
  area.red={model={}, feli={}, kyle={}, rj={}, sophia={}}
  for key, _ in pairs(f) do
    if key ~= "Mouse7R Scramble 10x 6t6" and key ~= "Mouse2L pro miR34a" then
      local values = file[key]
      table.insert(area.yellow.model, values.predictyellow:sum())
      table.insert(area.red.model, values.predictred:eq(2):sum())
      for _, person in pairs(persons) do
        table.insert(area.yellow[person], values[person]:eq(1):sum())
        table.insert(area.red[person], values[person]:eq(2):sum())
      end
    end
  end
  torch.save(segmentareafile, area)
end

local ratiofile = "interperson/areaRatio.t7"
local function computeAreaRatio()
  local retina = torch.load(retinafilename)
  local seg = torch.load(segmentareafile)
  local ratio = {yellow={}, red={}}
  local persons = {'model', 'feli', 'kyle', 'rj', 'sophia'}
  for _, person in pairs(persons) do
    ratio.yellow[person] = torch.cdiv(torch.Tensor(seg.yellow[person]), torch.Tensor(retina[person]))
    ratio.red[person] = torch.cdiv(torch.Tensor(seg.red[person]), torch.Tensor(retina[person]))
  end
  torch.save(ratiofile, ratio)
end

local function correlation(x, y)
  local n = x:size(1)
  local xy = torch.cmul(x,y)
  local x2 = torch.cmul(x,x)
  local y2 = torch.cmul(y,y)
  return (n*xy:sum() - x:sum()*y:sum()) / (math.sqrt(n*x2:sum()-x:sum()*x:sum())*math.sqrt(n*y2:sum()-y:sum()*y:sum()))
end

local function computeCorrelation()
  local ratio = torch.load(ratiofile)
  local truth = {'feli', 'kyle', 'rj', 'sophia'}
  local compare = {'feli', 'kyle', 'rj', 'sophia', 'model'}
  for _, gt in pairs(truth) do
    local x_yellow = ratio.yellow[gt]
    local x_red = ratio.red[gt]
    for _, person in pairs(compare) do
      if person ~= gt then
        local y_yellow = ratio.yellow[person]
        local y_red = ratio.red[person]
        print(string.format("gt: %s, %s, yellow: %f, red: %f", gt, person, correlation(x_yellow, y_yellow), correlation(x_red, y_red)))
      end
    end
  end  
end

local retinaDice = "interperson/retinaDice.t7"
local function computeRetinaDice()
  local retinaTensor = torch.load(retinaTensorName)
  local truth = {'feli', 'kyle', 'rj', 'sophia'}
  local compare = {'feli', 'kyle', 'rj', 'sophia', 'model'}
  local dice = {}
  for _, gt in pairs(truth) do
    dice[gt] = {}
    local x = retinaTensor[gt]
    for _, person in pairs(compare) do
      if gt ~= person then
        dice[gt][person] = torch.Tensor(#x)
        local y = retinaTensor[person]
        for i=1, #x do
          dice[gt][person][i] = diceCoef(y[i], x[i], 1)
        end
      end
    end
  end
  torch.save(retinaDice, dice)
end

local function saveRetinaDiceToFile()
  saveToFile(retinaDice, 'retina')
end

--createFileList(fileName)
--print(rawCnt)
--process()
--computeDice()
--plotDiceHist()
--plotDiceCompare()
--plotLabeledImage()
--saveDiceToFile()
--calMeanMedian("yellow")
--calMeanMedian("red")
--learnRetina()
--computeSegmentationArea()
--computeAreaRatio()
--computeCorrelation()
computeRetinaDice()
saveRetinaDiceToFile()
--newRetinaSegment()
