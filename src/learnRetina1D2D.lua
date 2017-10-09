require 'image'
require 'gnuplot'
require 'lfs'
local gm = require 'graphicsmagick'
local mlutils = require 'src/mlutils'
local utils = require 'src/utils'

local forward2ndDeri = false
local kstart = 1
local function secondDerivative(x)
 local rlt = x.new():resize(x:size(1)-2)
  for i=2, x:size(1)-1 do
    rlt[i-1] = x[i-1]+x[i+1]-2*x[i]
  end
  return rlt
end

local function secondDerivativeForward(x)
  local rlt = x.new():resize(x:size(1)-2)
  for i=1, rlt:size(1) do
    rlt[i] = x[i+2]-2*x[i+1]+x[i]
  end
  return rlt
end

local function learnRetina(filepath, d)
local basename = paths.basename(filepath, 'tif')
local img2D = gm.Image(filepath):toTensor('byte','RGB','DHW')[1]
local dsize = 256
local dimg = image.scale(img2D, dsize, dsize)
dimg = dimg:view(dsize*dsize):float()
local threshold = 100 -- 100 -- dimg:mean()
print("threshold", threshold)
local maskltTh = dimg:lt(threshold)
local dimg_vec = dimg:maskedSelect(maskltTh)
local dimg_vec_log = (dimg_vec+1):log()
print(dimg_vec_log:size())

local grid_y = torch.ger( torch.linspace(-1,1,dsize), torch.ones(dsize) )
local grid_x = torch.ger( torch.ones(dsize), torch.linspace(-1,1,dsize) )
local radius2_xy = torch.Tensor(dsize, dsize)
for i=1, dsize do
  for j=1, dsize do
    radius2_xy[i][j] = grid_x[i][j]*grid_x[i][j] + grid_y[i][j]*grid_y[i][j]
  end
end
radius2_xy_vec = radius2_xy:view(dsize*dsize)
radius2_xy_vec = radius2_xy_vec:maskedSelect(maskltTh)

gnuplot.pngfigure(string.format("/home/saxiao/tmp/retina2d/%s_scatter.png", basename))
gnuplot.plot(dimg_vec_log, radius2_xy_vec, '+')
gnuplot.plotflush()

local ks, losses = {}, {}
for k=kstart, 7 do
table.insert(ks, k)
local nIter = 10
local opt = {verbose=false}
local x
if d == 1 then
  x = dimg_vec_log:clone():view(-1,1)
else
  x = dimg_vec_log.new():resize(dimg_vec_log:size(1), 2)
  x[{{},1}]:copy(dimg_vec_log)
  x[{{},2}]:copy(radius2_xy_vec)
end
local m, label, counts, loss = mlutils.kmeans(x,k,nIter, nil, opt.kmeansCallback, opt.verbose)
table.insert(losses, loss)
print(m)

local p, retinaLabel = m[{{},1}]:max(1)
retinaLabel = retinaLabel:squeeze()
print("retinaLabel", retinaLabel)
local retina = torch.ByteTensor(dsize, dsize):fill(1)
retina:maskedCopy(maskltTh, label:eq(retinaLabel))

local plotname = string.format("/home/saxiao/tmp/retina2d/%s_retina_k%d.png", basename, k)
utils.drawImage(plotname, dimg:view(dsize, dsize):byte(), retina)
end

print(torch.Tensor(losses))
local elbow
if forward2ndDeri then
  local deri = secondDerivativeForward(torch.Tensor(losses))
  print(deri)
  _, elbow = deri:max(1)
  elbow = elbow[1] + kstart - 1
else
  local deri = secondDerivative(torch.Tensor(losses))
  print(deri)
  _, elbow = deri:max(1)
  elbow = elbow[1] + kstart
end
print("elbow", elbow)
gnuplot.pngfigure(string.format("/home/saxiao/tmp/retina2d/%s_loss_elbow_%d.png", basename, elbow))
gnuplot.plot(torch.Tensor(ks), torch.Tensor(losses))
gnuplot.plotflush()
end

local dir = "/data/oir-test/OIR quantification for Felicitas/IfnG P17 inj P14/20170206"
d=1
for file in lfs.dir(dir) do
  if string.match(file, 'tif') then
    print(file)
    local filepath = string.format("%s/%s", dir, file)
    learnRetina(filepath, d)
  end
end
