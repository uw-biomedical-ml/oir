
local split = "test"
local dir = "/home/saxiao/oir/data/"
local rawFile = torch.load(string.format("%s%s_raw.t7", dir, split))
local labelFile = torch.load(string.format("%s%s_label.t7", dir, split))
local data = {}
for i, raw in pairs(rawFile) do
  data[i] = {raw = raw, label = labelFile[i]}
end
local outputFile = string.format("%s%s_path.t7", dir, split)
torch.save(outputFile, data)
