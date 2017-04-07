require 'torch'
require 'lsf'
require 'paths'

local gm = require 'graphicsmagick'

local function getPathRecursively(dir, labeled)
  lfs.chdir(dir)

  for file in lfs.dir(dir) do
    if file ~= "." and file ~= ".." then
      local path = dir .. "/" .. file
      if lfs.attributes(path, "mode") == "directory" then
        getAllPaths(path, labeled)
      elseif lfs.attributes(path, "mode") == "file" then
        if string.match(file, "quantified") then
          table.insert(labeled,path)
	end
      end
    end
  end
end

local function getRawFilePath(labeledFile)
  local qStart = string.find(labeledFile, "quantified")
  local rawFile = string.sub(labeledFile,1,qStart-1)
  rawFile = string.gsub(rawFile, "%s*$", "") .. ".tif"
  return rawFile
end

local function retrieveAllPaths(pathsFile, dir)
  local labeled = {}
  getPathRecursively(dir, labeled)
  local cleanedLabeled = {}
  for _, file in pairs(labeled) do
    local rawFile = getRawFilePath(file)
     if paths.filep(rawFile) then
       table.insert(cleanedLabeled, file)
     end
end


