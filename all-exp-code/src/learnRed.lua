require 'cunn'
require 'cutorch'
cutorch.setDevice(1) -- note +1 to make it 0 indexed! sigh lua
cutorch.manualSeed(123)
require 'model'

local utils = require 'utils'

local dataDir = "/home/saxiao/oir/data/res256/train.t7"
local data = torch.load(dataDir)
local targetLabel = 2
local plotDir = "/home/saxiao/oir/plot/red/kmeans/"

local id = 1
local input, target = data[id].input, data[id].target
local originalType = input:type()
input, target = input:cuda(), target:cuda()
local redlabel = target:eq(targetLabel)
--utils.pixelHist(input, redlabel, string.format("%s/%d_hist.png", plotDir, id), {nbins=256, min=0, max=255, threshold=50})
utils.drawImage(string.format("%s/%d_t.png", plotDir, id), data[id].input, data[id].target)
utils.drawImage(string.format("%s/%d_r.png", plotDir, id), data[id].input)

local checkpoint = torch.load("/home/saxiao/oir/checkpoint/res256/augment/online/yellow/epoch_150.t7")
local net = checkpoint.model
net = net:cuda()
input, target = input:view(1,1, input:size(1), -1):cuda(), target:view(1, target:size(1), -1):cuda()
local output = net:forward(input)
local _, predictYellow = output:max(2)
predictYellow = predictYellow:squeeze()
predictYellow = predictYellow:view(redlabel:size(1), -1)
utils.drawImage(string.format("%s/%d_p_y.png", plotDir, id), input[1][1]:type(originalType), predictYellow:eq(2):type(originalType))

print(redlabel:sum())
redlabel:cmul(predictYellow:eq(1))
utils.pixelHist(input:type(originalType), redlabel:type(originalType), string.format("%s/%d", plotDir, id), {nbins=256, min=0, max=255, threshold=50, drawbg=true}) 
