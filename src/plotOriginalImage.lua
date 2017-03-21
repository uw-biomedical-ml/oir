require 'image'
local utils = require 'src/utils'

local filedir = "plot/yellow/oldrun/test/sorted"
local id = 197
local rawfile = string.format("%s/epoch_150_%d_r_original.png", filedir, id)
local labelfile = string.format("%s/epoch_150_%d_t_original.png", filedir, id)

local gm = require 'graphicsmagick'
local raw = gm.Image(rawfile):toTensor('byte','RGB','DHW')
local labelimg = gm.Image(labelfile):toTensor('byte','RGB','DHW')
print(labelimg:size())

local label = utils.getLabel(raw, labelimg, false)
local plotname = string.format("%s/epoch_150_%d_t.png", filedir, id)
utils.drawImage(plotname, raw[1], label:eq(1))
