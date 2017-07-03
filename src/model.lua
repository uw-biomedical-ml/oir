require 'torch'
require 'nn'
require 'nngraph'

local model = {}

local function convModule(input, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, depth)
  if not depth then depth = 2 end
  local convInput = input
  if torch.type(input) == "table" then
    convInput = input - nn.JoinTable(2)
  end
  local c = convInput
  local n1, n2 = nInputPlane, nOutputPlane
  for i = 1, depth do
    c = c
        - nn.SpatialConvolutionMM(n1, n2,kW,kH,dW,dH,padW,padH)
        - nn.SpatialBatchNormalization(nOutputPlane)
        - nn.ReLU()
    n1 = n2
  end
  return c
end

local function convModuleOld(input, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  local convInput = input
  if torch.type(input) == "table" then
    convInput = input - nn.JoinTable(2)
  end
  local c = convInput
	    - nn.SpatialConvolutionMM(nInputPlane, nOutputPlane,kW,kH,dW,dH,padW,padH)
	    - nn.SpatialBatchNormalization(nOutputPlane)
            - nn.ReLU()
	    - nn.SpatialConvolutionMM(nOutputPlane, nOutputPlane,kW,kH,dW,dH,padW,padH)
            - nn.SpatialBatchNormalization(nOutputPlane)
   	    - nn.ReLU()
  return c
end

local function convModule1(input, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  local convInput = input
  if torch.type(input) == "table" then
    convInput = input - nn.JoinTable(2)
  end
  local c = convInput
            - nn.SpatialConvolutionMM(nInputPlane, nOutputPlane,kW,kH,dW,dH,padW,padH)
            - nn.SpatialBatchNormalization(nOutputPlane)
            - nn.ReLU()
  return c
end

-- biggest receptive field: 140x140
function model.uNet1(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,16,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,16,32,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,32,32,3,3,1,1,1,1) -- 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 64x64
  local c4 = convModule(pool3,32,64,3,3,1,1,1,1) -- 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 136x136
  local c5 = convModule(pool4,64,64,3,3,1,1,1,1) -- 140x140
  
  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2)
  local c4Mirror = convModule({up1,c4},64+64,64,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2)
  local c3Mirror = convModule({up2,c3},64+32,32,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up3,c2},32+32,32,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up4,c1},32+16,16,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(16,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g 
end

function model.uNet1WithLocation(opt)
  local imgInput = - nn.Identity()  -- output: 1x256x256
  -- contracting path
  local c1 = convModule(imgInput,1,16,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2), output: 16x256x256
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10, output: 16x128x128
  local c2 = convModule(pool1,16,32,3,3,1,1,1,1) -- receptive field: 14x14, output: 32x128x128
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28, output: 32x64x64
  local c3 = convModule(pool2,32,32,3,3,1,1,1,1) -- 32x32, output: 32x64x64
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 64x64, output: 32x32x32
  local c4 = convModule(pool3,32,64,3,3,1,1,1,1) -- 68x68, output: 64x32x32
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 136x136, output: 64x16x16
  local c5 = convModule(pool4,64,64,3,3,1,1,1,1) -- 140x140, output: 64x16x16

  local locationInput = - nn.Identity() -- output: 2
  local l1 = locationInput - nn.Linear(2, 64) - nn.ReLU()  -- output 64
  local l2 = l1 - nn.Linear(64,64) - nn.ReLU() -- output: 64
  local locationOutput = l2 - nn.Replicate(16,2,1) - nn.Replicate(16,3,2) -- output: 64x16

  -- combine encoder output and location output
  local madd = nn.CAddTable()({c5, locationOutput})
  
  -- expansive path
  local up1 = madd - nn.SpatialUpSamplingNearest(2)
  local c4Mirror = convModule({up1,c4},64+64,64,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2)
  local c3Mirror = convModule({up2,c3},64+32,32,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up3,c2},32+32,32,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up4,c1},32+16,16,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(16,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({imgInput, locationInput},{last})
  return g 
end

-- receptive field 32x32
-- seems working well for segmenting yellow, and red, but haven't trained for a long till converging, so not sure
-- whether the best accuracy/dice it can achieve is as good.
function model.uNet2(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,16,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,16,32,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,32,32,3,3,1,1,1,1) -- 32x32

  -- expansive path
  local up1 = c3 - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up1,c2},32+32,32,3,3,1,1,1,1)
  local up2 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up2,c1},32+16,16,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(16,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

-- receptive field 32x32
function model.uNet3(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,8,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,8,16,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,16,16,3,3,1,1,1,1) -- 32x32

  -- expansive path
  local up1 = c3 - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up1,c2},16+16,16,3,3,1,1,1,1)
  local up2 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up2,c1},16+8,8,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(8,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

-- too simple for classifying red, trainDC at epoch 3000 is only 0.63
function model.uNet4(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule1(input,1,8,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule1(pool1,8,8,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule1(pool2,8,8,3,3,1,1,1,1) -- 32x32

  -- expansive path
  local up1 = c3 - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule1({up1,c2},8+8,8,3,3,1,1,1,1)
  local up2 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule1({up2,c1},8+8,8,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(8,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

-- unet5
-- classfigying red, train dice at epoch 3000 = 0.69
function model.uNet5(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,8,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,8,8,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,8,8,3,3,1,1,1,1) -- 32x32

  -- expansive path
  local up1 = c3 - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up1,c2},8+8,8,3,3,1,1,1,1)
  local up2 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up2,c1},8+8,8,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(8,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

--receptive field 140x140
-- uNet6
function model.uNet6(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule1(input,1,8,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule1(pool1,8,8,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule1(pool2,8,8,3,3,1,1,1,1) -- 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 64x64
  local c4 = convModule1(pool3,8,8,3,3,1,1,1,1) -- 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 136x136
  local c5 = convModule1(pool4,8,8,3,3,1,1,1,1) -- 140x140

  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2)
  local c4Mirror = convModule1({up1,c4},8+8,8,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2)
  local c3Mirror = convModule1({up2,c3},8+8,8,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule1({up3,c2},8+8,8,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule1({up4,c1},8+8,8,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(8,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

-- uNet7
function model.uNet7(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,8,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,8,8,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,8,8,3,3,1,1,1,1) -- 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 64x64
  local c4 = convModule(pool3,8,8,3,3,1,1,1,1) -- 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 136x136
  local c5 = convModule(pool4,8,8,3,3,1,1,1,1) -- 140x140

  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2)
  local c4Mirror = convModule({up1,c4},8+8,8,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2)
  local c3Mirror = convModule({up2,c3},8+8,8,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up3,c2},8+8,8,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up4,c1},8+8,8,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(8,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

-- uNet8
-- This is more complicated (deeper) than the baseline: uNet1
function model.uNet8(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,16,3,3,1,1,1,1,4)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,16,32,3,3,1,1,1,1,4) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,32,32,3,3,1,1,1,1,4) -- 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 64x64
  local c4 = convModule(pool3,32,64,3,3,1,1,1,1,4) -- 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 136x136
  local c5 = convModule(pool4,64,64,3,3,1,1,1,1,4) -- 140x140

  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2)
  local c4Mirror = convModule({up1,c4},64+64,64,3,3,1,1,1,1,4)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2)
  local c3Mirror = convModule({up2,c3},64+32,32,3,3,1,1,1,1,4)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up3,c2},32+32,32,3,3,1,1,1,1,4)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up4,c1},32+16,16,3,3,1,1,1,1,4)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(16,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

-- input 256 x 256, output is 8 times bigger, 2048 x 2048
-- jNet1
function model.jNet1(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,16,3,3,1,1,1,1)  -- output: 256x256, receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- output: 128x128, receptive field: 10x10
  local c2 = convModule(pool1,16,32,3,3,1,1,1,1) -- 128x128, receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- 64x64, receptive field: 28x28
  local c3 = convModule(pool2,32,32,3,3,1,1,1,1) -- 64x64, receptive field: 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 32x32, receptive field: 64x64
  local c4 = convModule(pool3,32,64,3,3,1,1,1,1) -- 32x32, receptive field: 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 16x16, receptive field: 136x136
  local c5 = convModule(pool4,64,64,3,3,1,1,1,1) -- 16x16, receptive field: 140x140

  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2) -- 32x32
  local c4Mirror = convModule({up1,c4},64+64,64,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2) -- 64x64
  local c3Mirror = convModule({up2,c3},64+32,32,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)  -- 128x128
  local c2Mirror = convModule({up3,c2},32+32,32,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)  -- 256x256
  local c1Mirror = convModule({up4,c1},32+16,16,3,3,1,1,1,1)
    
  local upj1 = c1Mirror - nn.SpatialUpSamplingNearest(2)  -- 512x512
  local cj1 = convModule1(upj1,16,16,3,3,1,1,1,1) -- 512x512
  local upj2 = cj1 - nn.SpatialUpSamplingNearest(2)  -- 1024x1024
  local cj2 = convModule1(upj2,16,16,3,3,1,1,1,1)  -- 1024x1024
  local upj3 = cj2 - nn.SpatialUpSamplingNearest(2)  -- 2048x2048
  local cj3 = convModule1(upj3,16,8,3,3,1,1,1,1)  -- 2048x2048

  -- make the right shape as the input
  local last = cj3
               - nn.SpatialConvolutionMM(8,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

-- jNet7
function model.jNet(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,8,3,3,1,1,1,1)  -- output: 256x256, receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- output: 128x128, receptive field: 10x10
  local c2 = convModule(pool1,8,8,3,3,1,1,1,1) -- 128x128, receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- 64x64, receptive field: 28x28
  local c3 = convModule(pool2,8,8,3,3,1,1,1,1) -- 64x64, receptive field: 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 32x32, receptive field: 64x64
  local c4 = convModule(pool3,8,8,3,3,1,1,1,1) -- 32x32, receptive field: 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 16x16, receptive field: 136x136
  local c5 = convModule(pool4,8,8,3,3,1,1,1,1) -- 16x16, receptive field: 140x140

  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2) -- 32x32
  local c4Mirror = convModule({up1,c4},8+8,8,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2) -- 64x64
  local c3Mirror = convModule({up2,c3},8+8,8,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)  -- 128x128
  local c2Mirror = convModule({up3,c2},8+8,8,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)  -- 256x256
  local c1Mirror = convModule({up4,c1},8+8,8,3,3,1,1,1,1)

  local upj1 = c1Mirror - nn.SpatialUpSamplingNearest(2)  -- 512x512
  local cj1 = convModule1(upj1,8,8,3,3,1,1,1,1) -- 512x512
  local upj2 = cj1 - nn.SpatialUpSamplingNearest(2)  -- 1024x1024
  local cj2 = convModule1(upj2,8,8,3,3,1,1,1,1)  -- 1024x1024
  local upj3 = cj2 - nn.SpatialUpSamplingNearest(2)  -- 2048x2048
  local cj3 = convModule1(upj3,8,8,3,3,1,1,1,1)  -- 2048x2048

  -- make the right shape as the input
  local last = cj3
               - nn.SpatialConvolutionMM(8,opt.nClasses,1,1,1,1,0,0)
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end


return model

