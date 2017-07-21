-- given an input image, predict labels and plot the prediction 
require 'nn'
require 'nngraph'
require 'image'

local gm = require 'graphicsmagick'
local utils = require 'utils'

local cmd = torch.CmdLine()
cmd:option('--imageFile', '../image/raw.tif', 'image file')
cmd:option('--target', "both", 'yellow|red|both')
cmd:option('--redModel', '../model/red.t7', 'trained model for red')
cmd:option('--yellowModel', '../model/yellow.t7', 'trained model for yellow')
cmd:option('--outputdir', '../output', 'output directory')
cmd:option('--learnRetina', true, 'whether to draw retina')
cmd:option('--verbose', true)
cmd:option('--gpu', 0, '-1 means using cpu, for i >= 0 means using gpu with id = i+1')
local opt = cmd:parse(arg)

local dtype = 'torch.DoubleTensor'
if opt.gpu >= 0 then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu+1) 
  cutorch.manualSeed(123)
  dtype = 'torch.CudaTensor'
end

paths.mkdir(opt.outputdir)

local red = {name="red", targetLabel=2, modelFile=opt.redModel, dsize=512}
local yellow = {name="yellow", targetLabel=1, modelFile=opt.yellowModel, dsize=256}
local retinaSize = 256

local img2D = gm.Image(opt.imageFile):toTensor('byte','RGB','DHW')[1]
local upPredict = torch.Tensor(img2D:size(1), img2D:size(2)):zero()

local function loadModel(modelFile)
  local model = torch.load(modelFile)
  model = model:type(dtype)
  return model
end

local function inference(modelFile, dsize)
  local model = loadModel(modelFile)
  model:evaluate()
  
  local dimg = image.scale(img2D, dsize, dsize)
  local originalType = dimg:type()
  dimg = dimg:type(dtype)
  local output = model:forward(dimg:view(1,1,dsize,-1))
  return output  
end

local function predict(targets)
  local retina = nil
  if opt.learnRetina then
    local dimg = image.scale(img2D, retinaSize, retinaSize)
    retina = utils.learnByKmeansThreshold(dimg, {verbose=opt.verbose})
    utils.drawImage(string.format("%s/retina.png", opt.outputdir), dimg, retina)
  end  
  for _, target in pairs(targets) do
    local output = inference(target.modelFile, target.dsize)
    local upsampled = utils.upsampleLabel(output:view(target.dsize,target.dsize,-1), {h=img2D:size(1), w=img2D:size(2)}) - 1
    upPredict:maskedFill(upsampled:eq(1), target.targetLabel)
    if retina then
      local _, predict = output:max(2)
      predict = predict - 1
      print(string.format("%s area ratio / retian = %.3f", target.name, predict:sum()/(retina:sum()*(target.dsize/retinaSize)^2)))
    end
  end
  utils.drawImage(string.format("%s/predict.png", opt.outputdir), img2D, upPredict)  
end

local targets = {}
if opt.target == 'yellow' then
  table.insert(targets, yellow)
elseif opt.target == 'red' then
  table.insert(targets, red)
elseif opt.target == 'both' then
  table.insert(targets, red)
  table.insert(targets, yellow)
end
predict(targets)
