-- given an input image, predict labels and plot the prediction 
require 'doPredict'

local cmd = torch.CmdLine()
cmd:option('--imageFile', 'sample/raw.png', 'image file')
cmd:option('--task', 3, '1(yellow)|2(red)|3(both)')
cmd:option('--redModel', 'model/red.t7', 'trained model for red')
cmd:option('--yellowModel', 'model/yellow.t7', 'trained model for yellow')
cmd:option('--retinaModel', 'model/retina.t7')
cmd:option('--outputdir', 'output', 'output directory')
cmd:option('--nnRetina', 1, 'set 1 means to use the nn model to predict the whole retina')
cmd:option('--kmeansRetina', 0, 'set 1 means to use kmeans to predict the whole retina')
cmd:option('--verbose', true)
cmd:option('--gpu', -1, '-1 means using cpu, for i >= 0 means using gpu with id = i+1')
cmd:option('--thumbnailSize', -1, 'thumbnail size')
cmd:option('--uniqueName', 0, '1 means keeping the basename of the input image')
local opt = cmd:parse(arg)

predict(opt)
