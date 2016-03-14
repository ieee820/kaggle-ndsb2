require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path

require 'cutorch'
require 'cunn'
pcall(require, 'cudnn')
require './common'

local function prediction(csv, norm_param, models, valid_x, valid_tag, valid_id)
   local AUG_SIZE = augmentation_size()
   local fp = io.open(csv, "w")
   fp:write("Id")
   for i = 0, 599 do
      fp:write(",P" .. i)
   end
   fp:write("\n")
   for i = 1, #valid_x do
      local nd = torch.Tensor(2, CRPS_N):zero()
      local cdf = torch.Tensor(2, CRPS_N):zero()
      local values = torch.Tensor(2, #models, 2, #valid_x[i], AUG_SIZE):zero()
      local c = 0
      for j = 1, #valid_x[i] do
	 local image_feat, tag_feat = augmentation(valid_x[i][j], valid_tag[i][j])
	 tag_feat = tag_feat:cuda()
	 image_feat = image_feat:cuda()
	 for k = 1, #models do
	    values[1][k][1][j]:copy(models[k][1][1]:forward({image_feat, tag_feat}):float()):mul(CRPS_N)
	    models[k][1][1]:clearState()
	    values[1][k][2][j]:copy(models[k][1][2]:forward({image_feat, tag_feat}):float()):mul(CRPS_N)
	    models[k][1][2]:clearState()
	    values[2][k][1][j]:copy(models[k][2][1]:forward({image_feat, tag_feat}):float()):mul(CRPS_N)
	    models[k][2][1]:clearState()
	    values[2][k][2][j]:copy(models[k][2][2]:forward({image_feat, tag_feat}):float()):mul(CRPS_N)
	    models[k][2][2]:clearState()
	 end
      end
      for j = 1, 2 do
	 local u = values[j]:mean()
	 local std = values[j]:std()
	 local sigma = math.pow((std * norm_param[j][2] + norm_param[j][1]), 2)
	 for x = 0, CRPS_N - 1 do
	    nd[j][x+1] = 1.0 / math.sqrt(2 * math.pi * sigma) * math.exp(-((x - u) * (x - u)/(2 * sigma)))
	 end
	 local cdfv = 0
	 for k = 1, CRPS_N do
	    cdfv = cdfv + nd[j][k]
	    cdf[j][k] = cdfv
	 end
	 local maxv = cdf[j]:max()
	 if maxv == 0 then
	    print("warning maxv = 0", j, valid_id[i])
	    cdf[j]:zero()
	 else
	    cdf[j]:div(maxv)
	 end
      end
      label = {"_Systole", "_Diastole"}
      for j = 1, 2 do
	 fp:write(valid_id[i] .. label[j])
	 for k = 1, cdf[j]:size(1) do
	    fp:write(",")
	    fp:write(cdf[j][k])
	 end
	 fp:write("\n")
      end
      xlua.progress(i, #valid_x)
      collectgarbage()
   end
   fp:close()
   xlua.progress(#valid_x, #valid_x)
end

torch.setnumthreads(4)
torch.setdefaulttensortype("torch.FloatTensor")

local normal_param = torch.load("models/normal_param.t7")
local models = {}
local x1 = torch.load(string.format("./data/validate_sax_x_64_16.t7"))
local tag1 = torch.load(string.format("./data/validate_sax_tag_64_16.t7"))
rebuild_sax(x1, tag1, nil, 3, 16)
local x2 = torch.load(string.format("./data/validate_sax_x_64_12.t7"))
local tag2 = torch.load(string.format("./data/validate_sax_tag_64_12.t7"))
rebuild_sax(x2, tag2, nil, 3, 16)
local id = torch.load("./data/validate_sax_id.t7")

print("load")

local x, tag, y = merge_data(x1, x2, tag1, tag2, y1, y2)
for i = 71, 78 do
   local model1 = {torch.load(string.format("models/1_1_%d.t7", i)), 
		   torch.load(string.format("models/1_2_%d.t7", i))}
   model1[1]:cuda():evaluate()
   model1[2]:cuda():evaluate()
   local model2 = {torch.load(string.format("models/2_1_%d.t7", i)),
		   torch.load(string.format("models/2_2_%d.t7", i))}
   model2[1]:cuda():evaluate()
   model2[2]:cuda():evaluate()
   table.insert(models, {model1, model2})
end
prediction("submission.txt", 
	   normal_param,
	   models, x, tag, id)
