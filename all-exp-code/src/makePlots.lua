require 'torch'
require 'gnuplot'

local function plotFigure(opt)
  gnuplot.pngfigure(opt.fileName)
  gnuplot.title(opt.title)
  if opt.setYRange then
    gnuplot.raw(opt.setYRange)
  end
  gnuplot.raw(opt.plotCommand)
  if opt.setKey then
    gnuplot.raw(opt.setKey)
  end
  gnuplot.xlabel('epoch')
  gnuplot.plotflush()
end

local function makePlots(opt, fileNameId, dataFileName)
  opt.fileName = fileNameId .. "_err.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 with errorlines title 'train', '' using 1:4:5 with errorlines title 'validate', '' using 1:6:7 with errorlines title 'test'"
--  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 with errorlines title 'train', '' using 1:6:7 with errorlines title 'test'"
  plotFigure(opt)

  opt.fileName = fileNameId .. ".png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 title 'train', '' using 1:4:5 title 'validate', '' using 1:6:7 title 'test'"
--  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 title 'train', '' using 1:6:7 title 'test'"
  plotFigure(opt)

  opt.fileName = fileNameId .. "_train.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 with errorlines title 'train'"
  plotFigure(opt)

  opt.fileName = fileNameId .. "_validate.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:4:5 with errorlines title 'validate'"
  plotFigure(opt)

  opt.fileName = fileNameId .. "_test.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:6:7 with errorlines title 'test'"
  plotFigure(opt)
end

local opt = {}
opt.title = "dice coefficient"
local fileNameRoot = "/home/saxiao/oir/plot/res256/augment/online/yellow-control/"
local fileNameId = string.format("%sdice", fileNameRoot)  -- "res256/augment/11x/dice"
local dataFileName = "/home/saxiao/oir/evalResultControlYellow.txt"  -- "plotData_res256_11x_dc.txt"
opt.setYRange = "set yrange [0:1]"
opt.setKey = "set key right bottom"
makePlots(opt, fileNameId, dataFileName)

-- plot loss
opt = {}
opt.fileName = string.format("%sloss.png", fileNameRoot)
opt.title = "loss"
opt.setYRange = "set yrange [0:0.3]"
opt.plotCommand = "plot '" .. dataFileName .. "' using 1:8 title 'train', '' using 1:9 title 'validate', '' using 1:10 title 'test'"
plotFigure(opt)

--opt = {}
--opt.title = "loss"
--fileNameId = "res256/augment/online/loss"  -- "res256/augment/11x/loss"
--dataFileName = "plotData_res256_online_loss.txt"  -- "plotData_res256_11x_loss.txt"
--opt.setYRange = "set yrange [0:0.3]"
--makePlots(opt, fileNameId, dataFileName)

