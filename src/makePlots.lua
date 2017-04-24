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
  opt.fileName = "/home/saxiao/oir/plot/" .. fileNameId .. "_err.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 with errorlines title 'train', '' using 1:4:5 with errorlines title 'validate', '' using 1:6:7 with errorlines title 'test'"
  plotFigure(opt)

  opt.fileName = "/home/saxiao/oir/plot/" .. fileNameId .. ".png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 title 'train', '' using 1:4:5 title 'validate', '' using 1:6:7 title 'test'"
  plotFigure(opt)

  opt.fileName = "/home/saxiao/oir/plot/" .. fileNameId .. "_train.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:2:3 with errorlines title 'train'"
  plotFigure(opt)

  opt.fileName = "/home/saxiao/oir/plot/" .. fileNameId .. "_validate.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:4:5 with errorlines title 'validate'"
  plotFigure(opt)

  opt.fileName = "/home/saxiao/oir/plot/" .. fileNameId .. "_test.png"
  opt.plotCommand = "plot '" .. dataFileName .. "' using 1:6:7 with errorlines title 'test'"
  plotFigure(opt)
end

local opt = {}
opt.title = "dice coefficient"
local fileNameId = "res256/noWeighted/dice"  -- "res256/augment/11x/dice"
local dataFileName = "plotData_res256_noAugment_dc.txt"  -- "plotData_res256_11x_dc.txt"
opt.setYRange = "set yrange [0:1]"
opt.setKey = "set key right bottom"
makePlots(opt, fileNameId, dataFileName)


opt = {}
opt.title = "loss"
fileNameId = "res256/noWeighted/loss"  -- "res256/augment/11x/loss"
dataFileName = "plotData_res256_noAugment_loss.txt"  -- "plotData_res256_11x_loss.txt"
opt.setYRange = "set yrange [0:0.3]"
makePlots(opt, fileNameId, dataFileName)

