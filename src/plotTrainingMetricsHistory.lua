require 'gnuplot'

local function plotFigure(opt, axisType)
  gnuplot.pngfigure(opt.fileName)
  gnuplot.title(opt.title)
  if opt.setYRange then
    gnuplot.raw(opt.setYRange)
  end
  --gnuplot.raw("set datafile separator ','")
  if opt.setXRange then
    gnuplot.raw(opt.setXRange)
  end
  gnuplot.raw(opt.plotCommand)
  if opt.setKey then
    gnuplot.raw(opt.setKey)
  end
  gnuplot.xlabel('iterations')
  if axisType then
    if axisType == 1 then
      gnuplot.axis{0,'',0.02,0.1}
    elseif axisType == 2 then
      gnuplot.axis{0,'',0.2,0.9}
    end
  end
  gnuplot.plotflush()
end

local rootDir = "/home/saxiao/oir/plot/red/"
paths.mkdir(rootDir)

local opt = {}
local modelId = "res512"
local trainFileName = "/home/saxiao/oir/" .. modelId .. "_train.txt"
local valFileName = "/home/saxiao/oir/" ..modelId .. "_val.txt"

opt.fileName = string.format("%s/%s_loss_val.png", rootDir, modelId)
opt.title = "validate loss"
opt.plotCommand = "plot '" .. valFileName .. "' using 1:2 notitle"
plotFigure(opt)


opt.fileName = string.format("%s/%s_loss_train.png", rootDir, modelId)
opt.title = "train loss"
opt.plotCommand = "plot '" .. trainFileName .. "' using 1:2 notitle"
plotFigure(opt)

opt.fileName = string.format("%s/%s_dc_val.png", rootDir, modelId)
opt.title = "validate dice"
opt.plotCommand = "plot '" .. valFileName .. "' using 1:3 notitle"
plotFigure(opt)


opt.fileName = string.format("%s/%s_dc_train.png", rootDir, modelId)
opt.title = "train dice"
opt.plotCommand = "plot '" .. trainFileName .. "' using 1:3 notitle"
plotFigure(opt)
