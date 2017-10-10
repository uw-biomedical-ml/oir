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
cmd:option('--outputdir', 'output', 'output directory')
cmd:option('--learnRetina', true, 'whether to draw retina')
cmd:option('--verbose', true)
cmd:option('--gpu', 0, '-1 means using cpu, for i >= 0 means using gpu with id = i+1')
cmd:option('--thumbnailSize', -1, 'thumbnail size')
cmd:option('--ks', {2})
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

  local red = {name="red", targetLabel=2, modelFile=opt.redModel, dsize=512}
  local yellow = {name="yellow", targetLabel=1, modelFile=opt.yellowModel, dsize=256}
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
    local retina = nil
    if opt.learnRetina then
      local dimg = image.scale(img2D, retinaSize, retinaSize)
      retina = {}
      for _, k in pairs(opt.ks) do
        retina[k] = utils.learnByKmeansThreshold(dimg, {k=k, verbose=opt.verbose})
        utils.drawImage(string.format("%s/%s_retina_k%d.png", opt.outputdir, basename, k), dimg:byte(), retina[k])
      end
      if opt.retinaModel then
        local retina_output = inference(opt.retinaModel, retinaSize)
        local _, retina_nn = retina_output:max(2)
        retina_nn = retina_nn:squeeze():view(retinaSize, retinaSize) - 1
        utils.drawImage(string.format("%s/%s_retina_nn.png", opt.outputdir, basename), dimg:byte(), retina_nn:byte())
        retina['nn'] = retina_nn
      end
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
      ratio[i] = {}
      if retina then
        for rkey, retinaLabel in pairs(retina) do
          local _, predict = output:max(2)
          predict = predict - 1
          ratio[i][rkey] = predict:sum()/(retinaLabel:sum()*(task.dsize/retinaSize)^2)
          print(string.format("%s area / retina ratio = %.3f", task.name, ratio[i][rkey]))
        end
      end
    end
    utils.drawImage(string.format("%s/%s_quantified.png", opt.outputdir, basename), img2D:byte(), upPredict)
    if opt.thumbnailSize > 0 then
      local tbImg2D = image.scale(img2D, opt.thumbnailSize, opt.thumbnailSize)
      local tblabel = image.scale(upPredict, opt.thumbnailSize, opt.thumbnailSize, 'simple')
      utils.drawImage(string.format("%s/%s_thumbnail.png", opt.outputdir, basename), tbImg2D:byte())
      utils.drawImage(string.format("%s/%s_quantified_thumbnail.png", opt.outputdir, basename), tbImg2D:byte(), tblabel) 
    end
    return ratio
  end
  return dopredict()
end

predict(opt)
