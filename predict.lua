-- given an input image, predict labels and plot the prediction 
require 'nn'
require 'nngraph'
require 'image'

local utils = require 'src/utils'
local gm = require 'graphicsmagick'

local cmd = torch.CmdLine()
cmd:option('--imageFile', 'sample/raw.png', 'image file')
cmd:option('--task', 3, '1(yellow)|2(red)|3(both)')
cmd:option('--redModel', 'model/red.t7', 'trained model for red')
cmd:option('--yellowModel', 'model/yellow.t7', 'trained model for yellow')
cmd:option('--retinaModel', 'model/retina.t7')
cmd:option('--outputdir', 'output', 'output directory')
cmd:option('--nnRetina', true, 'whether to draw retina')
cmd:option('--kmeansRetina', false, 'whether to draw retina')
cmd:option('--verbose', true)
cmd:option('--gpu', 0, '-1 means using cpu, for i >= 0 means using gpu with id = i+1')
cmd:option('--thumbnailSize', -1, 'thumbnail size')
cmd:option('--retinaModel', 'model/retina.t7')
local opt = cmd:parse(arg)

function predict(opt)
  local img2D = nil
  local supportFormat = {{png=1, PNG=2, jpg=3, jpeg=4, JPG=5, JPEG=6}, {tif=1, tiff=2}}
  local suffix = string.match(opt.imageFile, ".*%.(.*)")
  local basename = paths.basename(opt.imageFile, suffix)
  if supportFormat[1][suffix] then
    img2D = image.load(opt.imageFile)[1] * 255
  elseif supportFormat[2][suffix] then
    --local gm = require 'graphicsmagick'
    img2D = gm.Image(opt.imageFile):toTensor('byte','RGB','DHW')[1]
  else
    error(string.format("image format %s is not supported", suffix))
  end

  local dtype = 'torch.DoubleTensor'
  if opt.gpu >= 0 then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.gpu+1) 
    cutorch.manualSeed(123)
    dtype = 'torch.CudaTensor'
  end

  paths.mkdir(opt.outputdir)

  local red = {name="NV", targetLabel=2, modelFile=opt.redModel, dsize=512}
  local yellow = {name="VO", targetLabel=1, modelFile=opt.yellowModel, dsize=256}
  local retinaSize = 256

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

  local function dopredict()
    local pixelCnt = {retina={}}
    local retina = {}
    local dimg = image.scale(img2D, retinaSize, retinaSize)
    if opt.nnRetina then
      local retina_output = inference(opt.retinaModel, retinaSize)
      local _, retinaLabel = retina_output:max(2)
      retinaLabel = retinaLabel:squeeze():view(retinaSize, retinaSize) - 1
      retina.nn = retinaLabel
      utils.drawImage(string.format("%s/%s_retina_nn.png", opt.outputdir, basename), dimg:byte(), retinaLabel:byte())
      local retina_upsampled = image.scale(retinaLabel:byte(), img2D:size(1), img2D:size(2), 'simple')
      pixelCnt.retina.nn = retina_upsampled:sum()
    end
    if opt.kmeansRetina then
      local retinaLabel = utils.learnByKmeansThreshold(dimg, {k=2, verbose=opt.verbose})
      retina.kmeans = retinaLabel
      utils.drawImage(string.format("%s/%s_retina_kmeans.png", opt.outputdir, basename), dimg:byte(), retinaLabel:byte())
      local retina_upsampled = image.scale(retinaLabel:byte(), img2D:size(1), img2D:size(2), 'simple')
      pixelCnt.retina.kmeans = retina_upsampled:sum()
    end

    local tasks = {}
    if opt.task == 1 then
      table.insert(tasks, yellow)
    elseif opt.task == 2 then
      table.insert(tasks, red)
    elseif opt.task == 3 then
      table.insert(tasks, yellow)
      table.insert(tasks, red)
    else
      error(string.format("invalid task: %d", opt.task))
    end
    local ratio = {}
    for i, task in pairs(tasks) do
      local output = inference(task.modelFile, task.dsize)
      local upsampled = utils.upsampleLabel(output:view(task.dsize,task.dsize,-1), {h=img2D:size(1), w=img2D:size(2)}) - 1
      upPredict:maskedFill(upsampled:eq(1), task.targetLabel)
      local _, predict = output:max(2)
      predict = predict - 1
      pixelCnt[task.name] = predict:sum() * (img2D:size(1)/task.dsize)^2
      ratio[task.name] = {}
      for rkey, retinaLabel in pairs(retina) do
        ratio[task.name][rkey] = predict:sum()/(retinaLabel:sum()*(task.dsize/retinaSize)^2)
        print(string.format("%s area / retina(%s) ratio = %.3f", task.name, rkey, ratio[task.name][rkey]))
        pixelCnt.retina[rkey] = retinaLabel:sum() * (img2D:size(1)/retinaSize)^2
      end
    end
    utils.drawImage(string.format("%s/%s_quantified.png", opt.outputdir, basename), img2D:byte(), upPredict)
    if opt.thumbnailSize > 0 then
      local tbImg2D = image.scale(img2D, opt.thumbnailSize, opt.thumbnailSize)
      local tblabel = image.scale(upPredict, opt.thumbnailSize, opt.thumbnailSize, 'simple')
      utils.drawImage(string.format("%s/%s_thumbnail.png", opt.outputdir, basename), tbImg2D:byte())
      utils.drawImage(string.format("%s/%s_quantified_thumbnail.png", opt.outputdir, basename), tbImg2D:byte(), tblabel) 
    end
    return ratio, pixelCnt
  end
  return dopredict()
end

predict(opt)
