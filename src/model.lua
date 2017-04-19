require 'torch'
require 'nn'
require 'nngraph'

local model = {}

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

-- biggest receptive field: 140x140
function model.uNet(opt)
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

return model
