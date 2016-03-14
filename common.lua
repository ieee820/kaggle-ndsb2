require 'cutorch'
require 'image'
require 'optim'
local iproc = require 'iproc'

local FRAMES = 16
local FEAT_N = 20
local AUG_SIZE = 8
local FILTER = "bicubic"
CRPS_N = 600

function flatten_data(x, tag, y)
   local new_x = {}
   local new_tag = {}
   local new_y = {}
   for i = 1, #x do
      for j = 1, #x[i] do
	 table.insert(new_x, x[i][j])
	 table.insert(new_tag, tag[i][j])
	 table.insert(new_y, y[i][j])
      end
   end
   return new_x, new_tag, new_y
end
function stratified_sample(x, tag, y, max_n)
   local new_x = {}
   local new_tag = {}
   local new_y = {}
   for i = 1, #x do
      local perm = torch.randperm(#x[i])
      local n = math.min(#x[i], max_n)
      for j = 1, n do
	 table.insert(new_x, x[i][perm[j]])
	 table.insert(new_tag, tag[i][perm[j]])
	 table.insert(new_y, y[i][perm[j]])
      end
   end
   return new_x, new_tag, new_y
end
function gen_circle_mask(x_size)
   local mask = torch.Tensor(1, x_size[2], x_size[3]):fill(1)
   local center = (x_size[2] + 1) / 2
   local limit =  center * 0.95
   for i = 1, x_size[2] do
      for j = 1, x_size[3] do
	 local dist = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
	 if dist >= limit then
	    mask[1][i][j] = 0
	 end
      end
   end
   local mask_multi = torch.Tensor(x_size[1], x_size[2], x_size[3])
   for i = 1, x_size[1] do
      mask_multi[i]:copy(mask)
   end
   return mask_multi
end
local g_mask = nil
function normalize(x)
   g_mask = g_mask or gen_circle_mask(x:size())
   local mean = x:mean()
   local std = x:std() + 1.0e-16
   x:add(-mean):div(std)
   x:clamp(-2.5, 2.5)
   mean = x:mean()
   std = x:std() + 1.0e-16
   x:add(-mean):div(std)
   x:cmul(g_mask)
   --image.display(x)
   --sys.sleep(10)
   return x
end

function random_transform(x, y, tag, y_sigma)
   local x_size = x:size()
   x = x:reshape(x_size[1] * x_size[2], x_size[3], x_size[4])
   local theta = torch.uniform(0.0, 2 * math.pi)
   local scale = torch.normal(1.0, y_sigma)

   x = iproc.rotate(x, theta, FILTER)
   x = normalize(x)
   --image.display(x)
   --sys.sleep(10)
   x = x:reshape(x_size[1], x_size[2], x_size[3], x_size[4])
   local y1 = math.floor(y[1] * scale^3 * 600 + 0.5)
   local y2 = math.floor(y[2] * scale^3 * 600 + 0.5)
   if 1 <= y1 and y1 <= 600 and 1 <= y2 and y2 <= 600 then
      y = y:clone():mul(scale^3)
      tag = make_tag_feat(tag, scale)
   else
      tag = make_tag_feat(tag, 1.0)
   end
   return x, y, tag
end
function augmentation_size(use_scale)
   return AUG_SIZE
end
function augmentation(x, tag, use_scale)
   local x_size = x:size()
   local xs, tags

   xs = torch.Tensor(AUG_SIZE, 16, x_size[2], x_size[3], x_size[4])
   tags = torch.Tensor(AUG_SIZE, FEAT_N)

   local c = 1
   local o1;
   o1 = x
   o1 = o1:reshape(o1:size(1) * o1:size(2), o1:size(3), o1:size(4)):contiguous()
   for j = 1, 8 do
      local o2 = iproc.rotate(o1, (2 * math.pi) / 8 * (j - 1), FILTER)
      xs[c]:copy(normalize(o2))
      tags[c]:copy(make_tag_feat(tag, 1.0))
      c = c + 1
   end

   return xs, tags
end
function make_tag_feat(t, scale)
   scale = scale or 1.0
   local feat = torch.Tensor(FEAT_N):zero()
   local thick = tonumber(t["Slice Thickness"])
   if thick <= 8 then
      feat[1] = 1
      feat[2] = 0
   else
      feat[1] = 0
      feat[2] = 1
   end
   feat[3] = (t["_slice_distance"] or 10) / 10
   --20, 10, 5, 4
   feat[4] = tonumber(t["_scale_row"] * scale) / 4
   feat[5] = tonumber(t["_scale_col"] * scale) / 4
   feat[6] = math.log(tonumber(t["_scale_row"] * scale))
   feat[7] = math.log(tonumber(t["_scale_col"] * scale))
   feat[8] = feat[4] * feat[5]
   feat[9] = feat[3] * feat[4] * feat[5]
   feat[10] = math.log(feat[9])
   feat[11] = t["_slice_length"] / 100.0
   local slice_index = math.min(t["_slice_index"], 9)
   feat[11 + slice_index] = 1
   return feat
end
function optimize(model, y_index, criterion,
		  train_x, train_tag, train_y,
		  config)
   train_x, train_tag, train_y = stratified_sample(train_x, train_tag, train_y, 8)
   local x_size = train_x[1]:size()
   local parameters, gradParameters = model:getParameters()
   config = config or {}
   local sum_loss = 0
   local rmse = 0
   local count_loss = 0
   local shuffle = torch.randperm(#train_x)
   local num
   local c = 1
   local batch_size = config.xBatchSize or 32
   local block_size = config.xBlockSize or 32

   collectgarbage()

   if batch_size % block_size ~= 0 then
      error("xBatchSize % xBlockSize = must be 0")
   end
   for t = 1, #train_x, batch_size do
      if t + batch_size > #train_x then
	 break
      end
      xlua.progress(t, #train_x)
      local inputs1 = torch.Tensor(batch_size,
				   16,
				   --x_size[1],
				   x_size[2],
				   x_size[3],
				   x_size[4])
      local inputs2 = torch.Tensor(batch_size,
				   FEAT_N)
      local targets = torch.Tensor(batch_size, 1)
      for i = 1, batch_size do
	 local index = shuffle[t + i - 1]
	 local image_feat, scaled_y, tag_feat = random_transform(train_x[index], train_y[index], train_tag[index], config.xYSigma)
         inputs1[i]:copy(image_feat)
         inputs2[i]:copy(tag_feat)
	 targets[i][1] = scaled_y[y_index]
      end
      inputs1 = inputs1:cuda()
      inputs2 = inputs2:cuda()
      targets = targets:cuda()
      local feval = function(x)
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 gradParameters:zero()
	 local f = 0
	 for j = 1, batch_size, block_size do
	    local input = {inputs1[{{j, j + block_size - 1}}], inputs2[{{j, j + block_size - 1}}]}
	    local target = targets[{{j, j + block_size - 1}}]
	    local output = model:forward(input)
	    f = f + criterion:forward(output, target)
	    count_loss = count_loss + 1
	    rmse = rmse + math.sqrt((output - target):pow(2):sum() / output:nElement())
	    sum_loss = sum_loss + f
	    model:backward(input, criterion:backward(output, target))
	 end
	 return f, gradParameters
      end
      optim.adam(feval, parameters, config)
      c = c + 1
      if c % 4 == 0 then
	 collectgarbage()
      end
   end
   xlua.progress(#train_x, #train_x)
   return { huber = sum_loss / count_loss, rmse =  rmse / count_loss}
end
function remove_duplicated_slice(x, tag, y)
   for i = 1, #x do
      local slices = {}
      for j = 1, #x[i] do
	 table.insert(slices, {
			 slice = tonumber(tag[i][j]["Slice Location"]),
			 index = j,
			 time = tonumber(tag[i][j]["Acquisition Time"])
	 })
      end
      local selected = {}
      for j = 1, #slices do
	 local key = math.floor(math.floor(slices[j].slice) / 2)
	 if selected[key] then
	    if selected[key].time < slices[j].time then
	       selected[key] = slices[j]
	    end
	 else
	    selected[key] = slices[j]
	 end
      end
      local selected_array = {}
      for k, v in pairs(selected) do
	 table.insert(selected_array, v)
      end
      selected = selected_array
      table.sort(selected, function (a, b) return a.slice > b.slice end)
      local new_x = {}
      local new_y = {}
      local new_tag = {}
      for j = 1, #selected do
	 new_x[j] = x[i][selected[j].index]
	 if y then
	    new_y[j] = y[i][selected[j].index]
	 end
	 new_tag[j] = tag[i][selected[j].index]
	 new_tag[j]["_slice_length"] = tag[i][selected[1].index]["Slice Location"] - tag[i][selected[#selected].index]["Slice Location"]
      end
      x[i] = new_x
      if y then
	 y[i] = new_y
      end
      tag[i] = new_tag
   end
end
local function blend_slices(x, tag, y, saxes)
   local x_size = x[1][1]:size()
   for i = 1, #x do
      if i % 10 == 0 then
	 collectgarbage()
      end
      if #x[i] < saxes then
	 local mx = torch.Tensor(saxes, x_size[1], x_size[2], x_size[3])
	 for j = 1, #x[i] do
	    mx[j]:copy(x[i][j])
	 end
	 for j = #x[i] + 1, saxes do
	    mx[j]:copy(x[i][#x[i]])
	 end
	 x[i] = {mx:transpose(1, 2):contiguous()}
	 tag[i][j]["_slice_distance"] = (tonumber(tag[i][j]["Slice Location"]) - tonumber(tag[i][#x[i]]["Slice Location"])) / (#x[i] - 1)
	 tag[i][j]["_slice_index"] = 1
	 tag[i] = {tag[i][j]}
	 if y then
	    y[i] = {y[i][j]}
	 end
      else
	 local new_x = {}
	 local new_tag = {}
	 local new_y = {}
	 local n = #x[i] - saxes + 1
	 local s = 1
	 local e = n
	 if n > 4 then
	    -- drop first slice and last slice
	    s = 3
	    e = math.max(n - 3, s + 2)
	 end
	 c = 1
	 for j = s, e do
	    local mx = torch.Tensor(saxes, x_size[1], x_size[2], x_size[3])
	    for k = 1, saxes do
	       mx[k]:copy(x[i][j + (k - 1)])
	    end
	    mx = mx:transpose(1, 2):contiguous()
	    new_x[c] = mx
	    new_tag[c] = tag[i][j]
	    -- calc slice distance
	    new_tag[c]["_slice_distance"] = (tonumber(tag[i][j]["Slice Location"]) - tonumber(tag[i][j + (saxes - 1)]["Slice Location"])) / (saxes - 1)
	    new_tag[c]["_slice_index"] = (j - s) + 1
	    if y then
	       new_y[c] = y[i][j]
	    end
	    c = c + 1
	 end
	 x[i] = new_x
	 tag[i] = new_tag
	 if y then
	    y[i] = new_y
	 end
      end
   end
end
function truncate_frame(x, frame)
   local x_size = x[1][1]:size()
   for i = 1, #x do
      for j = 1, #x[i] do
	 local df = torch.Tensor(frame, x_size[2], x_size[3])
	 local lx = x[i][j]
	 for k = 1, frame do
	    df[k]:copy(lx[k + 1])
	 end
	 x[i][j] = df
      end
      collectgarbage()
   end
end

function rebuild_sax(x, tag, y, saxes, frame)
   truncate_frame(x, frame)
   blend_slices(x, tag, y, saxes)
end

function merge_data(x1, x2, tag1, tag2, y1, y2)
   --print(#x1, #x2, #tag1, #tag2, #y1, #y2)
   local x = {}
   local tag = {}
   local y = {}
   for i = 1, #x1 do
      local lx = {}
      local ltag = {}
      local ly = {}
      if #x1 ~= #x2 then
	 error("#x1 and #x2 missmatched")
      end
      for j = 1, #x1[i] do
	 table.insert(lx, x1[i][j])
	 table.insert(lx, x2[i][j])
	 table.insert(ltag, tag1[i][j])
	 table.insert(ltag, tag2[i][j])
	 if y1 and y2 then
	    table.insert(ly, y1[i][j])
	    table.insert(ly, y2[i][j])
	 end
      end
      x[i] = lx
      tag[i] = ltag
      y[i] = ly
   end
   --print(#x, #tag, #y)
   return x, tag, y
end
function crps(cdf, y)
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
function split_data(x, tag, y, test_size)
   local index = torch.randperm(#x)
   local train_size = #x - test_size
   local train_x = {}
   local train_tag = {}
   local train_y = {}
   local valid_x = {}
   local valid_tag = {}
   local valid_y = {}
   for i = 1, train_size do
      train_x[i] = x[index[i]]
      train_tag[i] = tag[index[i]]
      train_y[i] = y[index[i]]
  end
   for i = 1, test_size do
      valid_x[i] = x[index[train_size + i]]
      valid_tag[i] = tag[index[train_size + i]]
      valid_y[i] = y[index[train_size + i]]
   end
   return train_x, train_tag, train_y, valid_x, valid_tag, valid_y
end
function normalize_y(y)
   for i = 1, #y do
      for j = 1, #y[i] do
	 y[i][j]:div(CRPS_N)
      end
   end
   return y
end
