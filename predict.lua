-- given an input image, predict labels and plot the prediction 
require 'nn'
require 'nngraph'
require 'image'

local rootdir = "./"
package.path = package.path .. ";" .. rootdir .. "src/?.lua"

local utils = require 'utils'
local gm = require 'graphicsmagick'

local cmd = torch.CmdLine()
cmd:option('--imageFile', rootdir .. 'sample/raw.png', 'image file')
cmd:option('--task', 3, '1(yellow)|2(red)|3(both)')
cmd:option('--redModel', rootdir .. 'model/red.t7', 'trained model for red')
cmd:option('--yellowModel', rootdir .. 'model/yellow.t7', 'trained model for yellow')
cmd:option('--retinaModel', rootdir .. 'model/retina.t7')
cmd:option('--outputdir', rootdir .. 'output', 'output directory')
cmd:option('--nnRetina', 1, 'whether to draw retina')
cmd:option('--kmeansRetina', 0, 'whether to draw retina')
cmd:option('--verbose', true)
cmd:option('--gpu', -1, '-1 means using cpu, for i >= 0 means using gpu with id = i+1')
cmd:option('--thumbnailSize', -1, 'thumbnail size')
local opt = cmd:parse(arg)

function predict(opt)
  local function myerrorhandler(err)
    local errmsg = {}
    errmsg.error = "inference failed"
    errmsg.errordetail = err
    local jsonFile = string.format("%s/ratio.json", opt.outputdir)
    utils.write_json(jsonFile, errmsg)
  end

  timer = torch.Timer()
  local img3D, img2D = nil, nil
  
  local function loadImage()
    local supportFormat = {{png=1, PNG=2, jpg=3, jpeg=4, JPG=5, JPEG=6}, {tif=1, tiff=2, TIF=3, TIFF=4}}
    local suffix = string.match(opt.imageFile, ".*%.(.*)")
    if supportFormat[1][suffix] then
      img3D = image.load(opt.imageFile) * 255
    elseif supportFormat[2][suffix] then
      --local gm = require 'graphicsmagick'
      img3D = gm.Image(opt.imageFile):toTensor('byte','RGB','DHW')
    else
      error(string.format("image format %s is not supported", suffix))
    end
  end

  local status = xpcall(loadImage, myerrorhandler)
  if not status then
    print(status)
    return status
  end

  if img3D:size(1) == 3  then
    local _, channel = img3D:view(3,-1):float():mean(2):squeeze():max(1)
    img2D = img3D[channel[1]]
  else
    img2D = img3D[1]
  end

  local basename = paths.basename(opt.imageFile, suffix)
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
    if opt.nnRetina > 0 then
      local retina_output = inference(opt.retinaModel, retinaSize)
      local _, retinaLabel = retina_output:max(2)
      retinaLabel = retinaLabel:squeeze():view(retinaSize, retinaSize) - 1
      retina.nn = retinaLabel
      utils.drawImage(string.format("%s/retina.png", opt.outputdir), dimg:byte(), retinaLabel:byte())
    end
    if opt.kmeansRetina > 0 then
      local retinaLabel = utils.learnByKmeansThreshold(dimg, {k=2, verbose=opt.verbose})
      retina.kmeans = retinaLabel
      utils.drawImage(string.format("%s/retina.png", opt.outputdir), dimg:byte(), retinaLabel:byte())
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
    local ratio = {retina={}}
    for i, task in pairs(tasks) do
      local output = inference(task.modelFile, task.dsize)
      local upsampled = utils.upsampleLabel(output:view(task.dsize,task.dsize,-1), {h=img2D:size(1), w=img2D:size(2)}) - 1
      upPredict:maskedFill(upsampled:eq(1), task.targetLabel)
      local _, predict = output:max(2)
      predict = predict - 1
      ratio[task.name] = {}
      pixelCnt[task.name] = predict:sum() * (img2D:size(1)/task.dsize)^2
      ratio[task.name].pixel = pixelCnt[task.name]
      for rkey, retinaLabel in pairs(retina) do
        ratio[task.name][rkey] = predict:sum()/(retinaLabel:sum()*(task.dsize/retinaSize)^2)
        print(string.format("%s area / retina(%s) ratio = %.3f", task.name, rkey, ratio[task.name][rkey]))
        pixelCnt.retina[rkey] = retinaLabel:sum() * (img2D:size(1)/retinaSize)^2
        ratio.retina.pixel = pixelCnt.retina[rkey] 
      end
    end
    --utils.drawImage(string.format("%s/%s_quantified.done.png", opt.outputdir, basename), img2D:byte(), upPredict)
    utils.drawImage(string.format("%s/quantified.png", opt.outputdir), img2D:byte(), upPredict)
    if opt.thumbnailSize > 0 then
      local tbImg2D = image.scale(img2D, opt.thumbnailSize, opt.thumbnailSize)
      local tblabel = image.scale(upPredict, opt.thumbnailSize, opt.thumbnailSize, 'simple')
      utils.drawImage(string.format("%s/%s_thumbnail.png", opt.outputdir, basename), tbImg2D:byte())
      utils.drawImage(string.format("%s/%s_quantified_thumbnail.png", opt.outputdir, basename), tbImg2D:byte(), tblabel) 
    end
    --local ratioJsonFile = string.format("%s/%s_ratio.done.json", opt.outputdir, basename)
    local ratioJsonFile = string.format("%s/ratio.json", opt.outputdir)
    utils.write_json(ratioJsonFile, ratio)
    return ratio, pixelCnt
  end

  status, ratio, pixelCnt = xpcall(dopredict, myerrorhandler)
  print(status)
  return ratio, pixelCnt
end

predict(opt)
