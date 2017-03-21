require 'torch'
require 'gnuplot'

gnuplot.raw('set multiplot layout 2, 2')

local gm = require 'graphicsmagick'

local rawfile = "/home/saxiao/eclipse/workspace/oir/raw.tif"
local qfile = "/home/saxiao/eclipse/workspace/oir/quantified.tif"

local raw = gm.Image(rawfile):toTensor('byte','RGB','DHW')
local q = gm.Image(qfile):toTensor('byte','RGB','DHW')

local rsub1 = raw[1][{{1450,1500},{250,300}}]
local qsub1 = q[1][{{1450,1500},{250,300}}]

local rssub1 = rsub1[{{25,35},{25,35}}]
local qssub1 = qsub1[{{25,35},{25,35}}]

gnuplot.raw('set multiplot layout 2, 2')
gnuplot.imagesc(rsub1)
gnuplot.imagesc(qsub1)
gnuplot.imagesc(rssub1)
gnuplot.imagesc(qssub1)


