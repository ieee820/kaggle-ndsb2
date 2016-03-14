require 'pl'
require 'trepl'
require './common'

local function tries(results, bias, scale)
   local loss = 0
   local count = 0
   for i = 1, #results do
      for j = 1, #results[i] do
	 local u = results[i][j].mean
	 local std = results[i][j].std
	 local sigma = math.pow(std * scale + bias, 2)
	 local y = results[i][j].y
	 local cdf = torch.Tensor(CRPS_N):zero()
	 for x = 0, CRPS_N - 1 do
	    cdf[x+1] = 1.0 / math.sqrt(2 * math.pi * sigma) * math.exp(-((x - u) * (x  - u)/(2 * sigma)))
	 end
	 loss = loss + crps(cdf, y)
	 count = count + 1
      end
   end
   return loss / count
end

torch.setdefaulttensortype("torch.FloatTensor")
torch.manualSeed(71)
local normal_param = {}
local TRIES = 10000
for mode = 1, 2 do
   local results = {}
   for seed = 71, 78 do
      table.insert(results, torch.load(string.format("./models/results-%d_%d.t7", mode, seed)))
   end
   print("*******", mode)
   local best_score = tries(results, 0, 1, 1)
   local best_bias = 0
   local best_scale = 1
   print(best_score)
   for i = 1, TRIES do
      local bias = torch.uniform(0, 10)
      local scale = torch.uniform(0, 2)
      local score = tries(results, bias, scale)
      if score < best_score then
	 print(score, bias, scale)
	 best_score = score
	 best_bias = bias
	 best_scale = scale
      end
   end
   table.insert(normal_param, {best_bias, best_scale})
end
torch.save("models/normal_param.t7", normal_param)
