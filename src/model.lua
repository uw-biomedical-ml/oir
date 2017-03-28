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

function model.uNet(opt)
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
  local c2Mirror = {c3, c2}
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
