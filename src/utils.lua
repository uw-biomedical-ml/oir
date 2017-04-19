require 'image'

local utils = {}

function utils.drawImage(fileName, raw2D, label)
  local w, h = raw2D:size(1), raw2D:size(2)
  local img = raw2D.new():resize(3, w, h):zero()
  img[1]:copy(raw2D)
  if label then
    img[1]:maskedFill(label, 255)
    img[2]:maskedFill(label, 255)
  end
  image.save(fileName, img)
end

return utils
