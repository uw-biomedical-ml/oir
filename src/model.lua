require 'torch'
require 'nn'
require 'nngraph'

local model = {}

function model.uNetNoCopy(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = input
             - nn.SpatialConvolutionMM(1,64,3,3,1,1,1,1) -- 64 channels, (patchSize-3+1*2)/1+1 = patchSize, output 64 x patchSize x patchSize
             - nn.ReLU()
             - nn.SpatialConvolutionMM(64,64,3,3,1,1,1,1) -- 64 channels, output 64 x patchSize x patchSize
             - nn.ReLU()
  local c2 = c1
             - nn.SpatialMaxPooling(2,2)  -- output: 64 x patchSize/2 x patchSize/2
             - nn.SpatialConvolutionMM(64,128,3,3,1,1,1,1)  -- 128 channels, output 128 x patchSize/2 x patchSize/2
             - nn.ReLU()
             - nn.SpatialConvolutionMM(128,128,3,3,1,1,1,1)  -- 128 channels, output 128 x patchSize/2 x patchSize/2
             - nn.ReLU()
  local c3 = c2
             - nn.SpatialMaxPooling(2,2)  -- output: 128 x patchSize/4 x patchSize/4
             - nn.SpatialConvolutionMM(128,256,3,3,1,1,1,1)  -- 256 channels, output 256 x patchSize/4 x patchSize/4
             - nn.ReLU()
             - nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1)  -- 256 channels
             - nn.ReLU()
             - nn.SpatialUpSamplingNearest(2)  -- output: 256 x patchSize/2 x patchSize/2
  -- expansive path
  local c2Mirror = c3
                   - nn.SpatialConvolutionMM(256,128,3,3,1,1,1,1)  -- 128 filters, output 128 x patchSize/2 x patchSize/2
                   - nn.ReLU()
                   - nn.SpatialConvolutionMM(128,128,3,3,1,1,1,1)  -- 128 filters, output 128 x patchSize/2 x patchSize/2
                   - nn.ReLU()
                   - nn.SpatialUpSamplingNearest(2) -- output: 128 x patchSize x patchSize
  local c1Mirror = c2Mirror
                   - nn.SpatialConvolutionMM(128,64,3,3,1,1,1,1)  -- 64 filters, output 64 x patchSize x patchSize
                   - nn.ReLU()
                   - nn.SpatialConvolutionMM(64,64,3,3,1,1,1,1)  -- output 64 x patchSize x patchSize
                   - nn.ReLU()
  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(64,opt.nClasses,1,1,1,1,0,0) -- (patchSize-1)/1+1 = patchSize, output nClasses x patchSize x patchSize
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

local function convModule(input, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
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

function model.uNet(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,1,16,3,3,1,1,1,1)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)
  local c2 = convModule(pool1,16,32,3,3,1,1,1,1)
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)
  local c3 = convModule(pool2,32,32,3,3,1,1,1,1)
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)
  local c4 = convModule(pool3,32,64,3,3,1,1,1,1)
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)
  local c5 = convModule(pool4,64,64,3,3,1,1,1,1)
  
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

function model.uNetOld(opt)
  local input = - nn.Identity()
  -- contracting path
  local c1 = input
             - nn.SpatialConvolutionMM(1,64,3,3,1,1,1,1) -- receptive field: 3x3, 64 channels, (patchSize-3+1*2)/1+1 = patchSize, output 64 x patchSize x patchSize
             - nn.ReLU()
             - nn.SpatialConvolutionMM(64,64,3,3,1,1,1,1) -- receptive field: 5x5, 64 channels, output 64 x patchSize x patchSize
             - nn.ReLU()
  local c2 = c1
             - nn.SpatialMaxPooling(2,2)  -- receptive field: 10x10, output: 64 x patchSize/2 x patchSize/2
             - nn.SpatialConvolutionMM(64,128,3,3,1,1,1,1)  -- receptive field: 12x12, 128 channels, output 128 x patchSize/2 x patchSize/2
             - nn.ReLU()
             - nn.SpatialConvolutionMM(128,128,3,3,1,1,1,1)  -- receptive field: 14x14, 128 channels, output 128 x patchSize/2 x patchSize/2
             - nn.ReLU()
  local c3 = c2
             - nn.SpatialMaxPooling(2,2)  -- receptive field: 28x28, output: 128 x patchSize/4 x patchSize/4
             - nn.SpatialConvolutionMM(128,256,3,3,1,1,1,1)  -- receptive field: 30x30, 256 channels, output 256 x patchSize/4 x patchSize/4
             - nn.ReLU()
             - nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1)  -- receptive field: 32x32, 256 channels
             - nn.ReLU()
  local c4 = c3
             - nn.SpatialMaxPooling(2,2)  -- receptive field: 64x64, output: 128 x patchSize/8 x patchSize/8
             - nn.SpatialConvolutionMM(256,512,3,3,1,1,1,1)  -- receptive field: 66x66, 512 channels, output 512 x patchSize/8 x patchSize/8
             - nn.ReLU()
             - nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)  -- receptive field: 68x68, 512 channels
             - nn.ReLU()
             - nn.SpatialUpSamplingNearest(2)  -- output: 512 x patchSize/4 x patchSize/4
  -- expansive path
  local c3Mirror = {c4, c3}
                   - nn.JoinTable(2) -- output: (512+256) channels x patchSize/4 x patchSize/4
                   - nn.SpatialConvolutionMM(512+256,256,3,3,1,1,1,1)  -- 256 filters, output 256 x patchSize/4 x patchSize/4
                   - nn.ReLU()
                   - nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1)  -- 256 filters, output 256 x patchSize/4 x patchSize/4
                   - nn.ReLU()
                   - nn.SpatialUpSamplingNearest(2) -- output: 256 x patchSize/2 x patchSize/2
  local c2Mirror = {c3Mirror, c2}
                   - nn.JoinTable(2) -- output: (256+128) channels x patchSize/2 x patchSize/2
                   - nn.SpatialConvolutionMM(256+128,128,3,3,1,1,1,1)  -- 128 filters, output 128 x patchSize/2 x patchSize/2
                   - nn.ReLU()
                   - nn.SpatialConvolutionMM(128,128,3,3,1,1,1,1)  -- 128 filters, output 128 x patchSize/2 x patchSize/2
                   - nn.ReLU()
                   - nn.SpatialUpSamplingNearest(2) -- output: 128 x patchSize x patchSize
  local c1Mirror = {c2Mirror, c1}
                   - nn.JoinTable(2) -- output: (128+64) channels x patchSize x patchSize
                   - nn.SpatialConvolutionMM(128+64,64,3,3,1,1,1,1)  -- 64 filters, output 64 x patchSize x patchSize
                   - nn.ReLU()
                   - nn.SpatialConvolutionMM(64,64,3,3,1,1,1,1)  -- output 64 x patchSize x patchSize
                   - nn.ReLU()
  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(64,opt.nClasses,1,1,1,1,0,0) -- (patchSize-1)/1+1 = patchSize, output nClasses x patchSize x patchSize
               - nn.Reshape(opt.nClasses, -1)  -- output batchsize x nClasses x (number of pixels)
               - nn.Transpose({2,3})  -- output batchsize x (number of pixels) x nClasses
               - nn.Reshape(-1, opt.nClasses, false) -- output (batchsize* number of pixels) x nClasses
  local g = nn.gModule({input},{last})
  return g
end

return model
