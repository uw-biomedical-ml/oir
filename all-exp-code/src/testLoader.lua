require 'torch'

local opt = {}
opt.dir = '/Users/saxiao/AI/oir/data/'
opt.trainSize = 1
opt.validateSize = 0
opt.batchSize = 128
opt.patchSize = 10
opt.spacing = 4

local Loader = require 'Loader'
local loader = Loader.create(opt)

local function checkNumberOfIters(trainIter, expectedFileCursor)
  local imageSize = trainIter.getImageSize()
  local imageW, imageH = imageSize[1], imageSize[2]
  local expectedNumberOfIters = (((imageW-opt.patchSize)/opt.spacing + 1)*((imageH-opt.patchSize)/opt.spacing + 1))/opt.batchSize
  local cnt = 0
  while cnt < expectedNumberOfIters do
    trainIter.nextBatch()
    assert(trainIter.getFileCursor() == expectedFileCursor, "file cursor is wrong")
  end

end

local function testIterator()
  local trainIter = loader:iterator("train")
  local batch, label = trainIter.nextBatch()
  local firstBatch, firstLabel = batch:clone(), label:clone()
  assert(batch:size(1) == opt.batchSize, "batchSize is wrong")
  assert(batch:size(2) == opt.patchSize, "patchSize is wrong")
  assert(batch:size(3) == opt.patchSize, "patchSize is wrong")
  assert(label:size(1) == opt.batchSize, "label's size is wrong")
  trainIter.reset()
  checkNumberOfIters(trainIter, 1)
  checkNumberOfIters(trainIter, 2)
  batch, label = trainIter.nextBatch()
  local batchDiff = torch.eq(batch, firstBatch)
  assert(batchDiff:sum() == batchDiff:nElement(), "iterator should come back to the first plot, but it didn't")
  local labelDiff = torch.eq(label, firstLabel)
  assert(labelDiff:sum() == labelDiff:nElement(), "iterator should come back to the first plot, but it didn't")
end

testIterator()
