require 'gnuplot'
local mlutils = require 'src/mlutils'

local gm = require 'graphicsmagick'
local utils = require 'src/utils'
local dataDir = "/home/saxiao/oir/data/retina"
local plotDir = "/home/saxiao/oir/retina/downsample256log"
local downSample = true
local dH, dW = 256, 256
local idxstart, idxend = -1,-1
local reuseIndice = false
local useLog = true
local plotId = 1
local verbose = true
local printRaw = true

paths.mkdir(plotDir)

local function pixelHistRawLog(rawImg2D, plotName)
  --local nbins, min, max = 100, 1, 100
  local threshold = 100
  local maskltTh = rawImg2D:lt(threshold)
  local logImg = rawImg2D:maskedSelect(maskltTh):float()
  logImg = (logImg+1):log()

  --local logImg = (rawImg2D:float()+1):log()
  local nbins, min, max = 101, 0, math.log(100)
  local allhist = torch.histc(logImg, nbins, min, max)
  gnuplot.pngfigure(plotName)
  gnuplot.bar(allhist)
  gnuplot.plotflush()
end

local function pixelHistRaw(rawImg2D, plotName)
  local nbins, min, max = 100, 1, 100
  local allhist = torch.histc(rawImg2D:float(), nbins, min, max)
  gnuplot.pngfigure(plotName)
  gnuplot.bar(allhist)
  gnuplot.plotflush()
end

local function pixelHistRawZero(rawImg2D, plotName)
  local nbins, min, max = 101, 0, 100
  local allhist = torch.histc(rawImg2D:float(), nbins, min, max)
  gnuplot.pngfigure(plotName)
  gnuplot.bar(allhist)
  gnuplot.plotflush()
end

local function isRetinaBoundary(rgb)
  return rgb[3] > 200
end

local progress = {}
local function kmeansCallback(centroid, loss)
  print("centroids", centroid[1][1], centroid[2][1])
  print("loss", loss)
  table.insert(progress, {centroid=centroid, loss=loss})
end

local function learnByKmeansThreshold(img2D, plotName)
  -- 50 should be a hyper parameter
  local threshold = 100
  local maskltTh = img2D:lt(threshold)
  local x = img2D:maskedSelect(maskltTh):float()
  if useLog then
    x = (x+1):log()
  end
  x = x:view(-1,1)
  local nIter = 7 
  local k = 2
  progress = {}
  local m, label, counts = mlutils.kmeans(x,k,nIter, nil, kmeansCallback, verbose)
  print(counts)
  --print(label:min(), label:max())
  local retinaLabel = 1
  if m[1][1] < m[2][1] then retinaLabel = 2 end
  local imgLabel = torch.ByteTensor(img2D:size(1), img2D:size(2)):fill(1)
  imgLabel:maskedCopy(maskltTh, label:eq(retinaLabel))
  print(plotName)
  utils.drawRetina(plotName, img2D, imgLabel)
  return imgLabel
end

local function learnByKmeans(img2D, plotName)
  local x = img2D:view(-1,1)
  local nIter = 7
  local k = 2
  progress = {}
  local m, label, counts = mlutils.kmeans(x,k,nIter, nil, kmeansCallback, true)
  label = label:view(img2D:size(1), -1)
  print(counts)
  print(label:min(), label:max())
  local retinaLabel = 1
  if m[1][1] < m[2][1] then retinaLabel = 2 end
  utils.drawRetina(plotName, img2D, label:eq(retinaLabel))
end

local function learn()
  local file = torch.load("/home/saxiao/oir/data/test_raw.t7")
  local indice = nil
  local indiceFile = string.format("%s/indice.t7", plotDir)
  if reuseIndice then
    indice = torch.load(indiceFile)
  else
    if idxstart > 0 then
      if randomSample then
        indice = torch.randperm(#file)[{{idxstart, idxend}}]
      else
        indice = torch.range(1, #file)[{{idxstart, idxend}}]
      end
    else
      indice = torch.range(1, #file)
    end
    torch.save(indiceFile, indice)  
  end
  local allProgress = {}
  local idx = 0
  for i=1, indice:size(1) do
    idx = i
    if idxstart > 0 then
      idx = idx + idxstart - 1
    end   
    print(i)
    local rawImg = gm.Image(file[indice[i]]):toTensor('byte','RGB','DHW')
    local rawImg2D = rawImg[1]
    if downSample then
        rawImg2D = image.scale(rawImg2D, dH, dW)
    end
    local plotlogName = string.format("%s/%d_log.png", plotDir, idx)
    pixelHistRawLog(rawImg2D, plotlogName)
    local histRawZero = string.format("%s/%d.png", plotDir, idx)
    pixelHistRawZero(rawImg2D, histRawZero)
    if printRaw then
      local rawName = string.format("%s/%d_r.png", plotDir, idx)
      utils.drawImage(rawName, rawImg2D)
    end
    local learnPlot = string.format("%s/%d_kmeans.png", plotDir, idx)
    learnByKmeansThreshold(rawImg2D, learnPlot)
    table.insert(allProgress, progress)
  end
  if idxstart < 0 then
    torch.save(string.format("%s/progress.t7", plotDir), allProgress)
  end
end

local function evaluate()
  local N = 30
  local retinaDir = "/home/saxiao/oir/data/retina"
  local ratio = torch.Tensor(N)
  local eps = 1
  for i = 1, N do
    print(i)
    local file = torch.load(string.format("%s/%d.t7", retinaDir, i))
    local rawImg2D = file.input
    local trueLabel = file.target
    if downSample then
      rawImg2D = image.scale(rawImg2D, dH, dW)
      trueLabel = image.scale(trueLabel, dH, dW)
    end
    if printRaw then
      local rawName = string.format("%s/%d_r.png", plotDir, i)
      utils.drawImage(rawName, rawImg2D)
    end
    local learnPlot = string.format("%s/%d_kmeans.png", plotDir, i)
    local predictLabel = learnByKmeansThreshold(rawImg2D, learnPlot)
    ratio[i] = (predictLabel:sum()+eps) / (trueLabel:sum()+eps)
  end
  torch.save(string.format("%s/ratio.t7", plotDir), ratio)
end

local function ratioHist()
  local ratio = torch.load(string.format("%s/ratio.t7", plotDir))
  local ratioHistName = string.format("%s/hist.png", plotDir)
  gnuplot.pngfigure(ratioHistName)
  gnuplot.title("predict retina / true retina")
  gnuplot.hist(ratio, 40, 0, 2)
  gnuplot.plotflush()
end

local function ratioStat()
  local ratio = torch.load(string.format("%s/ratio.t7", plotDir))
  print("mean", ratio:mean())
  print("median", ratio:median())
  print("std", ratio:std())   
end

--learn()
evaluate()
ratioHist()
ratioStat()
