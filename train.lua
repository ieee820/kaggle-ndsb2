require 'pl'
require 'torch'
require 'cutorch'
require 'cunn'

local create_model = require './create_model'
require './HuberCriterion'
require './common'

local function crps(cdf, y)
   local output = torch.Tensor(CRPS_N):zero()
   local target = torch.Tensor(CRPS_N):zero()
   local cdfv = 0
   for i = 1, CRPS_N do
      cdfv = cdfv + cdf[i]
      output[i] = cdfv
      if y <= (i - 1) then
	 target[i] = 1
      else
	 target[i] = 0
      end
   end
   output:div(output:max() + 1.0e-16)
   --gnuplot.plot({'out', output, '-'}, {'y', target, '-'})
   local loss = (output - target):pow(2):sum() / CRPS_N
   return loss
end
local function test(model, y_index, criterion, valid_x, valid_tag, valid_y)
   local loss = 0
   local rmse = 0
   local AUG_SIZE = augmentation_size()
   local slice_count = {}
   local slice_rmse = {}
   for i = 1, #valid_x do
      local y = valid_y[i][1]:cuda()
      local cdf = torch.Tensor(600):zero()
      local values = torch.Tensor(#valid_x[i], AUG_SIZE):zero()
      for j = 1, #valid_x[i] do
	 local image_feat, tag_feat = augmentation(valid_x[i][j], valid_tag[i][j])
	 tag_feat = tag_feat:cuda()
	 image_feat = image_feat:cuda()
	 values[j]:copy(model:forward({image_feat, tag_feat}):float():mul(CRPS_N))
	 if not slice_count[j] then
	    slice_count[j] = 0
	    slice_rmse[j] = 0
	 end
	 slice_count[j] = slice_count[j] + 1
	 slice_rmse[j] = slice_rmse[j] + math.pow(y[y_index] *CRPS_N - values[j]:mean(), 2)
      end
      local u
      local sigma
      sigma = math.pow(values:std(), 2)
      u = values:mean()
      for x = 0, CRPS_N - 1 do
	 cdf[x+1] = 1.0 / math.sqrt(2 * math.pi * sigma) * math.exp(-((x - u) * (x  - u)/(2 * sigma)))
      end
      loss = loss + crps(cdf, y[y_index] * CRPS_N)
      rmse = rmse + math.pow(y[y_index] - u / CRPS_N, 2)
      xlua.progress(i, #valid_x)
      collectgarbage()
   end
   for k, c in pairs(slice_count) do
      print(k, c, math.sqrt(slice_rmse[k] / c) / CRPS_N)
   end
   xlua.progress(#valid_x, #valid_x)
   return loss / #valid_x, math.sqrt(rmse / #valid_x)
end
local function training(opt)
   cutorch.manualSeed(opt.seed)
   torch.manualSeed(opt.seed)
   local MAX_EPOCH = 150
   local best_score = 1000.0
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
   local model = create_model(opt.model)
   local criterion = HuberCriterion(0.02):cuda()
   local sgd_config = {
      learningRate = 0.0001,
      epsilon = 1.0e-16,
      xBatchSize = 40,
      xBlockSize = 20,
      xYSigma = opt.y_sigma
   }
   local label = {"systole", "diastole"}
   local wd = false
   for epoch = 1, MAX_EPOCH do
      print("# " .. epoch)
      print("## train " .. label[t])
      model:training()
      local train_score = optimize(model, t, criterion, train_x, train_tag, train_y, sgd_config)
      print(train_score)
      if train_score.rmse < 0.04 and epoch % 2 == 0 then
	 print("## test" .. label[t])
	 model:evaluate()
	 local score, rmse = test(model, t, criterion, valid_x, valid_tag, valid_y)
	 print({CRPS = score, RMSE = rmse, best = best_score})
	 if rmse < best_score then
	    best_score = rmse
	    torch.save(string.format("models/%d_%d_%d.t7", opt.mode, opt.model, opt.seed),
		       model:clearState())
	 end
      end
   end
end
local cmd = torch.CmdLine()
cmd:text()
cmd:text("Kaggle-BOWL2 SAX")
cmd:text("Options:")
cmd:option("-seed", 71, 'fixed input seed')
cmd:option("-mode", 1, '1(systole)|2(diastole)')
cmd:option("-model", 1, '1|2')
cmd:option("-y_sigma", 0.01, "random std for y")

local opt = cmd:parse(arg)
print(opt)
torch.setnumthreads(4)
torch.setdefaulttensortype("torch.FloatTensor")
training(opt)
