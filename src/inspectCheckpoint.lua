require 'torch'
require 'nn'
require 'nngraph'
require 'gnuplot'
require 'lfs'

local function plotFigure(fileName, title, x, y1, y2)
  gnuplot.pngfigure(fileName)
  gnuplot.title(title)
  if y2 then
    gnuplot.plot({'train', x, y1, '+'}, {'validate', x, y2, '+'})
  else
    gnuplot.plot({'train', x, y1, '+'})
  end
  
  gnuplot.plotflush()
end

local function plotProgress(dataDir, plotDir)
  local files = {}
  for file in lfs.dir(dataDir) do
    if lfs.attributes(dataDir .. file, "mode") == "file" then
      table.insert(files,dataDir .. file)
    end
  end
  -- sort files by epoch number
  local function findNumber(str)
    local i, j = str:find("/[^/]*$")
    return tonumber(str:sub(i+7, j-3))
  end

  table.sort(files, function (a,b) return findNumber(a) < findNumber(b) end)
  local epoch = torch.Tensor(#files)
  local loss = torch.Tensor(#files)
  local trainAccuracy = torch.Tensor(#files)
  local validateAccuracy = torch.Tensor(#files)
  local trainDiceCoef = torch.Tensor(#files)
  local validateDiceCoef = torch.Tensor(#files)

  for i = 1, #files do
    local checkpoint = torch.load(files[i])
    epoch[i] = checkpoint.epoch
    loss[i] = checkpoint.loss
    trainAccuracy[i] = checkpoint.trainAccuracy
    validateAccuracy[i] = checkpoint.validateAccuracy
    trainDiceCoef[i] = checkpoint.trainDiceCoef
    validateDiceCoef[i] = checkpoint.validateDiceCoef
  end
 
  plotFigure(string.format("%sloss.png",plotDir), "loss", epoch, loss)
  plotFigure(string.format("%saccuracy.png", plotDir), "accuracy", epoch, trainAccuracy, validateAccuracy)
  plotFigure(string.format("%sdiceCoef.png", plotDir), "dice coef", epoch, trainDiceCoef, validateDiceCoef)
end

local prefix = "/home/saxiao/oir/"
local checkpointDir = prefix .. "checkpoint/cntf/"
local plotDir = prefix .. "plot/"
plotProgress(checkpointDir, plotDir)
