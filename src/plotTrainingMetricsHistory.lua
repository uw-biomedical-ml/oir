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
      gnuplot.axis{0,'',0,0.5}  -- red {0,'',0.02,0.24}, yellow: {0,'',0.02,0.35}
    elseif axisType == 2 then
      gnuplot.axis{0,'',0.5,1}  -- red: {0,'',0,0.9}, yellow: {0,'',0,1}
    end
  end
  gnuplot.plotflush()
end

print("plotTrainingMetrics")

local rootDir = "./plot/retina"
paths.mkdir(rootDir)

local opt = {}
local modelId = "retina"
local trainFileName = "./" .. modelId .. "_train.txt"
local valFileName = "./" ..modelId .. "_val.txt"

opt.fileName = string.format("%s/%s_loss_val.png", rootDir, modelId)
opt.title = "validate loss"
opt.plotCommand = "plot '" .. valFileName .. "' using 1:2 with points notitle"
plotFigure(opt, 1)


opt.fileName = string.format("%s/%s_loss_train.png", rootDir, modelId)
opt.title = "train loss"
opt.plotCommand = "plot '" .. trainFileName .. "' using 1:2 with points notitle"
plotFigure(opt, 1)

opt.fileName = string.format("%s/%s_dc_val.png", rootDir, modelId)
opt.title = "validate dice"
opt.plotCommand = "plot '" .. valFileName .. "' using 1:3 with points notitle"
plotFigure(opt, 2)


opt.fileName = string.format("%s/%s_dc_train.png", rootDir, modelId)
opt.title = "train dice"
opt.plotCommand = "plot '" .. trainFileName .. "' using 1:3 with points notitle"
plotFigure(opt, 2)
