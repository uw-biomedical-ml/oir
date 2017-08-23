require 'image'
require 'gnuplot'
local mlutils = require 'src/mlutils'

local utils = {}

function utils.varyContrast(input, scale)
  input = input:float()
  local mean = input:mean()
  return input:mul(scale):add((1 - scale) * mean)
end

-- N classes, not including background
-- The first dimension is batchsize
function utils.diceCoef(predict, target, nClasses, isBatch)
  if not isBatch or predict:nDimension() == 1 then
    predict = predict:view(1, -1)
    target = target:view(1, -1)
  else
    local nBatches = predict:size(1)
    predict = predict:view(nBatches, -1)
    target = target:view(nBatches, -1)
  end

  local predictLabel = predict - 1
  local label = target - 1
  --local n = label:max()
  local a, b, c = 0, 0, 0
  local eps = 1
  for i = 1, nClasses do
    local pi = predictLabel:eq(i):float()
    local ti = label:eq(i):float()
    a = a + torch.cmul(pi, ti):sum(2)
    b = b + pi:sum(2)
    c = c + ti:sum(2)
  end
  local dice = torch.cdiv(2*a + eps, b + c + eps)
  return dice:squeeze()
end

function utils.drawRetina(fileName, img2D, label)
  local h, w = img2D:size(1), img2D:size(2)
  local img3D = img2D.new():resize(3, h, w):zero()
  img3D[1]:copy(img2D)
  img3D[1]:maskedFill(label, 255)
  img3D[2]:maskedFill(label, 255)
  image.save(fileName, img3D)
end

function utils.upsampleLabel(output, originalSize)
  local outputFloat = output:float()
  local nClasses = outputFloat:size(3)
  local upsampleOutput = outputFloat.new():resize(originalSize.w, originalSize.h, nClasses)
  for i = 1, nClasses do
    upsampleOutput[{{},{},i}]:copy(image.scale(outputFloat[{{},{},i}], originalSize.w, originalSize.h))
  end
  local _, upsamplePredict = upsampleOutput:max(3)
  upsamplePredict = upsamplePredict:squeeze()
  return upsamplePredict
end

function utils.drawImage(fileName, raw2D, label)
  local w, h = raw2D:size(1), raw2D:size(2)
  local img = raw2D.new():resize(3, w, h):zero()
  img[1]:copy(raw2D)
  if label then
    img[2]:copy(raw2D)
    img[3]:copy(raw2D)
    local yellowMask = label:eq(1)
    img[1]:maskedFill(yellowMask, 255)
    img[2]:maskedFill(yellowMask, 255)
    img[3]:maskedFill(yellowMask, 0)
    local redMask = label:eq(2)
    img[1]:maskedFill(redMask, 255)
    img[2]:maskedFill(redMask, 0)
    img[3]:maskedFill(redMask, 0)
    if label:max() == 3 then
      local retinaMask = label:eq(3)
      img[1]:maskedFill(retinaMask, 255)
      img[2]:maskedFill(retinaMask, 255)
      img[3]:maskedFill(retinaMask, 255)
    end
  end
  image.save(fileName, img)
end

local function isGrey(image, includeRetina)
  local m12 = image[1]:eq(image[2])
  local m23 = image[2]:eq(image[3])
  local m123 = torch.cmul(m12, m23)  -- bg=1, if grey scale: normal(pixels in all three channels are 0)=1, yellow, red=0, if not grey scale: normal/red=0, yellow=0
  local m1Not0 = image[1]:ne(0)
  local m123Not0 = torch.cmul(m123, m1Not0)
  local isGrey = false
  local labelMask = nil
  if m123Not0:sum() > 0 then
    isGrey =  true
    labelMask = m123 * (-1) + 1  -- flip 0's and 1's
    local rgDiff = (image[1]:float():add(-1, image[2]:float())):abs()
    local gbDiff = (image[2]:float():add(-1, image[3]:float())):abs()
    local redYellowCutoff = 30  -- 30
    local gbYellowCutoff = 30
    if rgDiff:max() > redYellowCutoff then
      -- There are red and yellow highlighted
      labelMask:zero()
      labelMask:maskedFill(rgDiff:ge(redYellowCutoff), 2)  -- red = 2
      labelMask:maskedFill(rgDiff:lt(redYellowCutoff):cmul(gbDiff:gt(gbYellowCutoff)), 1)  -- yellow + normal + bg = 1
      labelMask:maskedFill(m123, 0)  -- normal+bg=0
      if includeRetina then
        local rbDiff = (image[1]:float():add(-1, image[3]:float())):abs()
        labelMask:maskedFill(image[3]:gt(100):cmul(rbDiff:gt(30)), 3)
      end
    end
  end
  return isGrey, labelMask
end

function utils.getLabel(rawImage, labelImage, includeRetina)
  local label = rawImage[2]:ne(labelImage[2])  -- yellow = 1, the rest (normal, bg, red possibly) = 0
  local isGrey, qMask = isGrey(labelImage, includeRetina)
  if isGrey then
    label = qMask
  end
  return label
end

local function scanContour(imgDHW, isBoundaryFunc, backward)
  local h, w = imgDHW:size(2), imgDHW:size(3)
  local filled = imgDHW.new():resize(h, w)
  local from, to, step = 1, w, 1
  if backward then 
    from, to, step = w, 1, -1
  end
  for i = 1, h do
    local cnt = 0
    local isCurrentBoundary = false
    for j = from, to, step do
      if isBoundaryFunc(imgDHW[{{1,-1},i,j}]) then
        if not isCurrentBoundary then
          isCurrentBoundary = true
          cnt = cnt + 1
        end
      else
        if isCurrentBoundary then
          isCurrentBoundary = false
        end
      end
      filled[i][j] = cnt
    end
  end
  filled = filled % 2
  return filled
end

function utils.fillContour(imgDHW, isBoundaryFunc)
  local filledForward = scanContour(imgDHW, isBoundaryFunc)
  local filledBackward = scanContour(imgDHW, isBoundaryFunc, true)
  local filled = filledForward:cmul(filledBackward)
  return filled
end

local function fillDfs(imgDHW, isBoundaryFunc, visited, filled, h, w)
  if not isBoundaryFunc(imgDHW[{{1,3},h,w}]) and visited[h][w] == 0 then
    visited[h][w] = 1
    filled[h][w] = 1
    if h+1 <= filled:size(1) then
      fillDfs(imgDHW, isBoundaryFunc, visited, filled, h+1, w)
    end
    if h-1 > 0 then
      fillDfs(imgDHW, isBoundaryFunc, visited, filled, h-1, w)
    end
    if w+1 <= filled:size(2) then
      fillDfs(imgDHW, isBoundaryFunc, visited, filled, h, w+1)
    end
    if w-1 > 0 then
      fillDfs(imgDHW, isBoundaryFunc, visited, filled, h, w-1)
    end
  end
end

function utils.fillDfs(imgDHW, isBoundaryFunc)
  local H, W = imgDHW:size(2), imgDHW:size(3)
  local startH, startW = math.floor(H/2), math.floor(W/2)
  local visited = torch.ByteTensor(H,W):zero()
  local filled = torch.ByteTensor(H,W):zero()
  local stack = {}
  table.insert(stack, {startH, startW})
  while #stack > 0 do
    local p = table.remove(stack)
    local h, w = p[1], p[2]
    if not isBoundaryFunc(imgDHW[{{1,3},h,w}]) and visited[h][w] == 0 then
      visited[h][w] = 1
      filled[h][w] = 1
      if h+1 <= H then
        table.insert(stack, {h+1, w})
      end
      if h-1 > 0 then
        table.insert(stack, {h-1, w})
      end
      if w+1 <= W then
        table.insert(stack, {h, w+1})
      end
      if w-1 > 0 then
        table.insert(stack, {h, w-1})
      end
    end
  end
  --fillDfs(imgDHW, isBoundaryFunc, visited, filled, startH, startW)
  --print(filled:sum())
  return filled
end

function utils.pixelHist(rawImg2D, label, plotName, opt)
  local nbins, min, max = opt.nbins, opt.min, opt.max
  local threshold = opt.threshold
  rawImg2D = rawImg2D:float()
  local thresholdMask = torch.ByteTensor():resize(label:size(1), label:size(2)):fill(1)
  if threshold then
    thresholdMask = rawImg2D:gt(threshold)
  end
  print(label:sum())
  print(label:type(), thresholdMask:type())
  print(torch.cmul(label, thresholdMask):sum())
  local signalArea = rawImg2D:maskedSelect(torch.cmul(label, thresholdMask))
  local shist = torch.histc(signalArea, nbins, min, max)
  local bgArea = rawImg2D:maskedSelect(torch.cmul(label:eq(0), thresholdMask))
  local bghist = torch.histc(bgArea, nbins, min, max)
  local allArea = rawImg2D:maskedSelect(thresholdMask)
  local allhist = torch.histc(allArea, nbins, min, max)
  local fileName = "/home/saxiao/tmp/hist.txt"
  local f = io.open(fileName, 'w')
  for i=1, shist:size(1) do
    f:write(string.format("%d %d %d %d\n", i, shist[i], bghist[i], allhist[i]))
  end
  f:close()
  local signalPlot = string.format("%s_hist_s.png", plotName)
  gnuplot.pngfigure(signalPlot)
  gnuplot.raw("plot '" .. fileName .. "' using 1:2 with boxes title 'target'")
  gnuplot.plotflush()
  local bgplot = string.format("%s_hist_bg.png", plotName)
  gnuplot.pngfigure(bgplot)
  gnuplot.raw("plot '" .. fileName .. "' using 1:3 with boxes title 'bg'")
  gnuplot.plotflush()

--  gnuplot.raw("plot '" .. fileName .. "' using 1:2 with boxes title 'target', '' using 1:3 with boxes title 'background', '' using 1:4 with boxes title 'all'")
end

function utils.learnByKmeansThreshold(img2D, opt)
  if not opt then opt = {} end
  -- should be a hyper parameter
  local threshold = 100
  local maskltTh = img2D:lt(threshold)
  local x = img2D:maskedSelect(maskltTh):float()
  x = (x+1):log():view(-1,1)
  local nIter = 7
  local k = 2
  progress = {}
  local m, label, counts = mlutils.kmeans(x,k,nIter, nil, opt.kmeansCallback, opt.verbose)
  --print(counts)
  --print(label:min(), label:max())
  local retinaLabel = 1
  if m[1][1] < m[2][1] then retinaLabel = 2 end
  local imgLabel = torch.ByteTensor(img2D:size(1), img2D:size(2)):fill(1)
  imgLabel:maskedCopy(maskltTh, label:eq(retinaLabel))
  return imgLabel
end

function utils.scale(raw, label, w, h, plotOpt)
  local draw = nil
  if raw then
    draw = image.scale(raw, w, h)
  end
  local dlabel = nil
  if label then
    dlabel = label.new():resize(h, w):zero()
    for i = 1, label:max() do
      local imask = image.scale(label:eq(i), w, h)
      dlabel:maskedFill(imask:gt(0.5), i)
    end
  end
  if plotOpt then
    local imageFileRoot = string.format("%splot/%d", rootDir, plotOpt.idx)
    utils.drawImage(string.format("%s_t.png", imageFileRoot), draw, dlabel)
    local sortedOriginalLabel = string.format("%s_t_original.png", imageFileRoot)
    os.execute("cp \"" .. plotOpt.originalLabelFile .. "\" \"" .. sortedOriginalLabel .. "\"")
    utils.drawImage(string.format("%s_r.png", imageFileRoot), draw)
    local sortedOriginalRaw = string.format("%s_r_original.png", imageFileRoot)
    os.execute("cp \"" .. plotOpt.originalRawFile .. "\" \"" .. sortedOriginalRaw .. "\"")
  end

  return draw, dlabel
end

function utils.findMatchedRawFile(files, fileName, key, labelId)
  for file in lfs.dir(fileName) do
    if file ~= "." and file ~= ".." then
      local path = fileName .. "/" .. file
      if lfs.attributes(path, "mode") == "directory" then
        utils.findMatchedRawFile(files, path, key)
      elseif lfs.attributes(path, "mode") == "file" and string.match(file, key) then
        local qStart = string.find(file, key)
        local rawFilePrefix = string.sub(file,1,qStart-1)
        rawFilePrefix = string.gsub(rawFilePrefix, "%s*$", "")
        local tmpFile = "tmp.txt"
        os.execute("find /data/oir/ -name '*" .. rawFilePrefix .. "*' > " .. tmpFile)
        local file = io.open(tmpFile)
        for line in file:lines() do
          if not string.match(line, key) then
            table.insert(files, {raw=line, labelId=path})
            break
          end
        end
      end
    end
  end
end

return utils
