require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'torch'
require 'cutorch'
require 'cunn'
pcall(require, 'cudnn')
require './common'

local function test(models, y_index, valid_x, valid_tag, valid_y)
   local loss = 0
   local rmse = 0
   local AUG_SIZE = augmentation_size()
   local slice_count = {}
   local slice_rmse = {}
   local results = {}
   for i = 1, #valid_x do
      local y = valid_y[i][1]:cuda()
      local cdf = torch.Tensor(600):zero()
      local values = torch.Tensor(#valid_x[i], #models, AUG_SIZE):zero()
      for j = 1, #valid_x[i] do
	 local image_feat, tag_feat = augmentation(valid_x[i][j], valid_tag[i][j], SCALE_AUGMENT)
	 tag_feat = tag_feat:cuda()
	 image_feat = image_feat:cuda()
	 values[j][1]:copy(models[1]:forward({image_feat, tag_feat}):float():mul(CRPS_N))
	 values[j][2]:copy(models[2]:forward({image_feat, tag_feat}):float():mul(CRPS_N))
	 if not slice_count[j] then
	    slice_count[j] = 0
	    slice_rmse[j] = 0
	 end
	 slice_count[j] = slice_count[j] + 1
	 slice_rmse[j] = slice_rmse[j] + math.pow(y[y_index] *CRPS_N - values[j]:mean(), 2)
      end
      local u = values:mean()
      local std = values:std()
      local sigma = math.pow(std, 2)
      for x = 0, CRPS_N - 1 do
	 cdf[x+1] = 1.0 / math.sqrt(2 * math.pi * sigma) * math.exp(-((x - u) * (x  - u)/(2 * sigma)))
      end
      loss = loss + crps(cdf, y[y_index] * CRPS_N)
      rmse = rmse + math.pow(y[y_index] - u / CRPS_N, 2)
      table.insert(results, {
		      rmse = math.sqrt(math.pow(y[y_index] * CRPS_N - u, 2)),
		      mean = u,
		      std = std,
		      y = y[y_index] * CRPS_N, 
		      patient_id = valid_tag[i][1]["Patient ID"], 
      })
      models[1]:clearState()
      models[2]:clearState()
      xlua.progress(i, #valid_x)
      collectgarbage()
   end
   for k, c in pairs(slice_count) do
      print(k, c, math.sqrt(slice_rmse[k] / c) / CRPS_N)
   end
   table.sort(results, function (a, b) return a.rmse > b.rmse end)
   xlua.progress(#valid_x, #valid_x)
   return loss / #valid_x, math.sqrt(rmse / #valid_x), results
end
local function run(opt)
   cutorch.manualSeed(opt.seed)
   torch.manualSeed(opt.seed)
   local x1 = torch.load(string.format("./data/train_sax_x_64_16.t7"))
   local tag1 = torch.load(string.format("./data/train_sax_tag_64_16.t7"))
   local y1 = normalize_y(torch.load(string.format("./data/train_sax_y_64_16.t7")))
   rebuild_sax(x1, tag1, y1, 3, 16)
   local x2 = torch.load(string.format("./data/train_sax_x_64_12.t7"))
   local tag2 = torch.load(string.format("./data/train_sax_tag_64_12.t7"))
   local y2 = normalize_y(torch.load(string.format("./data/train_sax_y_64_12.t7")))
   rebuild_sax(x2, tag2, y2, 3, 16)

   local t = opt.mode

   print("load")

   local x, tag, y = merge_data(x1, x2, tag1, tag2, y1, y2)

   local train_x, train_tag, train_y, valid_x, valid_tag, valid_y = split_data(x, tag, y, 50)
   local models = {}
   models[1] = torch.load(string.format("models/%d_1_%d.t7", opt.mode, opt.seed)):cuda()
   models[2] = torch.load(string.format("models/%d_2_%d.t7", opt.mode, opt.seed)):cuda()
   models[1]:cuda():evaluate()
   models[2]:cuda():evaluate()
   local label = {"systole", "diastole"}
   local score, rmse, results = test(models, t, valid_x, valid_tag, valid_y)
   torch.save(string.format("models/results-%d_%d.t7", opt.mode, opt.seed), results)
   print({CRPS = score, RMSE = rmse})
end
local cmd = torch.CmdLine()
cmd:text()
cmd:text("Kaggle-BOWL2 SAX test")
cmd:text("Options:")
cmd:option("-seed", 71, 'fixed input seed')
cmd:option("-mode", 1, '1(systole)|2(diastole)')
local opt = cmd:parse(arg)
print(opt)
torch.setnumthreads(4)
torch.setdefaulttensortype("torch.FloatTensor")
run(opt)
