require 'torch'
require 'gnuplot'

local gm = require 'graphicsmagick'

local dir = "/Users/saxiao/AI/oir/"
local file = torch.load(dir .. "data/allPaths.t7")
local qfile = file[1]
local qStart = string.find(qfile, "quantified")
local rawfile = string.sub(qfile,1,qStart-1)
rawfile = string.gsub(rawfile, "%s$", "") .. ".tif"
print(rawfile) 

local raw = gm.Image(rawfile):toTensor('byte','RGB','DHW')
local q = gm.Image(qfile):toTensor('byte','RGB','DHW')
local labels = torch.eq(raw[2], q[2]) 

local w = raw:size(2)
local patchSize = 64
local n = 1180
local centerX = (patchSize * n - patchSize/2) % w 
local centerY = patchSize * ((patchSize * n - patchSize/2) / w + 1) - patchSize/2
print(centerX, centerY)

local rsub1 = raw[1][{{centerX-patchSize/2+1,centerX+patchSize/2},{centerY-patchSize/2+1,centerY+patchSize/2}}]
local qsub1 = labels[{{centerX-patchSize/2+1,centerX+patchSize/2},{centerY-patchSize/2+1,centerY+patchSize/2}}]

local savefile = dir .. "plot/raw" .. centerX .. "_" .. centerY .. ".png"
gnuplot.pngfigure(savefile)
gnuplot.imagesc(rsub1)
gnuplot.plotflush()

savefile = dir .. "plot/q" .. centerX .. "_" .. centerY .. ".png"
gnuplot.pngfigure(savefile)
gnuplot.imagesc(qsub1)
gnuplot.plotflush()

