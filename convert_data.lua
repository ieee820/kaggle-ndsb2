require 'pl'
require 'trepl'
package.path = path.join("..", "lib", "?.lua;") .. package.path

local csvigo = require 'csvigo'
local cjson = require 'cjson'
local gm = require 'graphicsmagick'
local image = require 'image'
local iproc = require 'iproc'
require 'cutorch'
require 'common'

torch.setdefaulttensortype("torch.FloatTensor")
torch.setnumthreads(2)

local function make_seq_image(list, tag)
   local frames = math.min(30, #list)
   local x = nil
   local step = frames / 30
   for i = 1, 30 do
      local index = math.floor(step * (i - 1) + 1)
      local im = gm.Image():load(list[index])
      local w, h = im:size()
      if w <= h then
	 im = im:rotate(360-90)
	 w, h = im:size()
      end
      if h / w > 0.75 then
	 -- fix ratio
	 local new_h = w * 0.75
	 im = im:crop(w, new_h, 0, (h - new_h) * 0.5)
	 w, h = im:size()
      end
      local hs = opt.image_size + opt.calibration_mergin * 2
      local ws = math.floor((hs / h) * w)
      if not x then
	 x = torch.Tensor(30, hs, ws):zero()
      end
      im = im:size(ws, hs, "Lanczos"):toTensor("float", "I", "DHW")
      x[i]:copy(im)
   end
   tag["_time_frames"] = frames
   local min = x:min()
   x:add(-min)
   --image.display(x)
   --sys.sleep(5)
   return x
end
local function get_center(fg4)
   local pos = {}
   for y = 1, fg4:size(2) do
      for x = 1, fg4:size(3) do
	 if fg4[1][y][x] > 0.0 then
	    table.insert(pos, {x = x, y = y, val4 = fg4[1][y][x]})
	 end
      end
   end
   local sum4 = 0
   for i = 1, #pos do
      sum4 = sum4 + pos[i].val4
   end
   local center_x = 0
   local center_y = 0
   for i = 1, #pos do
      center_x = center_x + pos[i].x * (pos[i].val4 / sum4)
      center_y = center_y + pos[i].y * (pos[i].val4 / sum4)
   end
   local crop_size = math.min(fg4:size(2), fg4:size(3)) * 0.75 * 0.5
   sum4 = 0
   for i = 1, #pos do
      if center_x - crop_size < pos[i].x and
	 pos[i].x < center_x + crop_size and 
	 center_y - crop_size < pos[i].y and
	 pos[i].y < center_y + crop_size then
	    sum4 = sum4 + pos[i].val4
      end
   end
   local center_fix_x = 0
   local center_fix_y = 0
   for i = 1, #pos do
      if center_x - crop_size < pos[i].x and
	 pos[i].x < center_x + crop_size and 
	 center_y - crop_size < pos[i].y and
	 pos[i].y < center_y + crop_size then
	    center_fix_x = center_fix_x + pos[i].x * (pos[i].val4 / sum4)
	    center_fix_y = center_fix_y + pos[i].y * (pos[i].val4 / sum4)
      end
   end
   return {center_fix_y, center_fix_x}
end
local function debug_disp(x, c)
   print(x:size(), c)
   local d = torch.Tensor(3, x:size(2), x:size(3))
   d[1]:copy(x)
   d[2]:copy(x)
   d[3]:copy(x)
   d:div(d:max())
   d[1][c[1]][c[2]] = 1.0
   d[1][c[1]+1][c[2]] = 1.0
   d[1][c[1]][c[2]+1] = 1.0
   d[1][c[1]+1][c[2]+1] = 1.0
   image.display(d)
end
local g_gauss = image.gaussian(3, 3)
local function calibrate_image(p_x, id)
   local x_size = p_x[1]:size()
   local fg = torch.Tensor(1, x_size[2], x_size[3]):zero()
   local smooth = {}
   local st = 1
   local ed = #p_x
   if #p_x > 5 then
      st = 2
      ed = #p_x - 1
   end
   for i = st, ed do
      local s = p_x[i]:clone()
      local mean = s:mean()
      local std = s:std()
      s:add(-mean):div(std+1.0e-16)
      s:clamp(-2.0, 2.0)
      for j = 1, 30 do
	 s[j]:copy(image.convolve(image.convolve(s[j]:reshape(1, s:size(2), s:size(3)), g_gauss, "same"), g_gauss, "same"))
      end
      if x_size[2] ~= s:size(2) or x_size[3] ~= s:size(3) then
	 print(id)
	 print(x_size[2], s:size(2), x_size[3], s:size(3))
	 error("fail")
      end
      fg:add(s:std(1):pow(4))
   end
   fg:clamp(fg:mean(), fg:max())
   local center = get_center(fg)
   local ey = math.min(center[1] + opt.image_size / 2, x_size[2])
   local sy = math.max(ey - opt.image_size, 0)
   local ew = math.min(center[2] + opt.image_size / 2, x_size[3])
   local sw = math.max(ew - opt.image_size, 0)
   
   --debug_disp(fg, center)
   --image.display(iproc.crop(p_x[3], sw, sy, sw + opt.image_size, sy + opt.image_size))
   --sys.sleep(2)
   for i = 1, #p_x do
      p_x[i] = iproc.crop(p_x[i], sw, sy, sw + opt.image_size, sy + opt.image_size)
   end
end
local function clearn_tag(tag)
   local new_tag = {}
   for k, v in pairs(tag) do
      if type(v) == "number" or type(v) == "string" then
	 new_tag[k] = v
      end
   end
   return new_tag
end
local function remove_size_mismatched(x, tag)
   local sizes = {}
   for i = 1, #x do
      local key = tag[i]["Columns"] .. "x" .. tag[i]["Rows"]
      if not sizes[key] then
	 sizes[key] = {{x[i], tag[i]}}
      else
	 table.insert(sizes[key], {x[i], tag[i]})
      end
   end
   local best = 0
   local selected = nil
   local c = 0
   for k, v in pairs(sizes) do
      if #v > best then
	 selected = v
	 best = #v
      end
      c = c + 1
   end
   if c > 1 then
      x = {}
      tag = {}
      for i = 1, #selected do
	 table.insert(x, selected[i][1])
	 table.insert(tag, selected[i][2])
      end
   end
   return x, tag
end
local function make_tag(js)
   local tag = clearn_tag(js)
   -- calc image scale
   local rows = tonumber(tag["Rows"])
   local cols = tonumber(tag["Columns"])
   local spacing = utils.split(tag["Pixel Spacing"], "\\")
   local spacing_row = tonumber(spacing[1])
   local spacing_col =  tonumber(spacing[2])
   local hs = opt.image_size + opt.calibration_mergin * 2
   if cols <= rows then
      local t = rows
      local t2 = spacing_row
      rows = cols
      cols = t
      spacing_row = spacing_col
      spacing_col = t2
   end
   if rows / cols > 0.75 then
      -- fix ratio
      rows = cols * 0.75
   end
   local ws = math.floor((hs / rows) * cols)
   --print(rows, cols, hs, ws, (rows / hs), (cols / ws))
   tag["_scale_row"] = spacing_row * (rows / hs)
   tag["_scale_col"] = spacing_col * (cols / ws)

   return tag
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
local function process_data(train_dir, labels)
   local train_x = {}
   local train_tag = {}
   local train_y = {}
   local train_id = {}
   local patients = dir.getdirectories(train_dir)
   local force_count = 0

   table.sort(patients, 
	      function (a, b) return tonumber(path.basename(a)) < tonumber(path.basename(b)) end)

   for i = 1, #patients do
      local p_x = {}
      local p_tag = {}
      local p_y = {}
      local patient_id = path.basename(patients[i])
      local y = nil

      if labels then
	 local label = labels[patient_id]
	 y = {label[1], label[2], label[3]}
      end
      local entries = dir.getdirectories(path.join(patients[i], "study"))
      table.sort(entries)
      for j = 1, #entries do
	 local sax = false
	 if entries[j]:match("sax_%d") then
	    sax = true
	 end
	 if sax then
	    local dcms = dir.getfiles(entries[j], "*.dcm")
	    local slices = {}
	    local tags = {}
	    for k = 1, #dcms do
	       --print(dcms[k]:gsub("dcm", "json"))
	       local js = cjson.decode(file.read(dcms[k]:gsub("dcm", "json")))
	       local key = js["Slice Location"]--js["Series Number"] .. "_" .. js["Slice Location"]
	       if slices[key] then
		  table.insert(slices[key], dcms[k])
	       else
		  slices[key] = {dcms[k]}
	       end
	       if not tags[key] then
		  tags[key] = make_tag(js)
	       end
	    end
	    for k, v in pairs(slices) do
	       -- ??
	       if #slices[k] == 30 then
		  -- ok
		  table.sort(slices[k])
		  table.insert(p_x, make_seq_image(slices[k], tags[k]))
		  table.insert(p_tag, tags[k])
		  if y then
		     table.insert(p_y, torch.Tensor(y))
		  end
	       elseif #slices[k] == 60 then
		  local new_slices = {}
		  local new_tags = {}
		  for l = 1, #slices[k] do
		     local new_key = (path.basename(slices[k][l]):split('%.'))[1]:split("-")[4]
		     local js = cjson.decode(file.read(slices[k][l]:gsub("dcm", "json")))
		     if new_slices[new_key] then
			table.insert(new_slices[new_key], slices[k][l])
		     else
			new_slices[new_key] = {slices[k][l]}
		     end
		     if not new_tags[key] then
			new_tags[new_key] = make_tag(js)
		     end
		  end
		  for lk, lv in  pairs(new_slices) do
		     if #new_slices[lk] == 30 then
			-- ok
			table.sort(new_slices[lk])
			table.insert(p_x, make_seq_image(new_slices[lk], new_tags[lk]))
			table.insert(p_tag, new_tags[lk])
			if y then
			   table.insert(p_y, torch.Tensor(y))
			end
		     else
			print("Failed to load", #slices[k], #new_slices[lk], entries[j])
		     end
		  end
	       else
		  -- force set
		  force_count = force_count + 1
		  table.sort(slices[k])
		  table.insert(p_x, make_seq_image(slices[k], tags[k]))
		  table.insert(p_tag, tags[k])
		  if y then
		     table.insert(p_y, torch.Tensor(y))
		  end
	       end
	    end
	 end
      end
      p_x, p_tag = remove_size_mismatched(p_x, p_tag)
      calibrate_image(p_x, patient_id)
      table.insert(train_x, p_x)
      table.insert(train_tag, p_tag)
      table.insert(train_y, p_y)
      table.insert(train_id, patient_id)

      xlua.progress(i, #patients)
      collectgarbage()
   end
   remove_duplicated_slice(train_x, train_tag, train_y)
   print("force count", force_count)
   return train_x, train_tag, train_y, train_id 
end
local function load_labels(label_file)
   local csv = csvigo.load({path = label_file, verbose = false, mode = "raw"})
   local labels = {}
   -- remove header
   table.remove(csv, 1)
   for i = 1, #csv do
      local patient_id = csv[i][1]
      local esv = tonumber(csv[i][2])
      local edv = tonumber(csv[i][3])
      labels[patient_id] = { esv, edv }
   end
   return labels
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("Kaggle-BOWL2 SAX Convert")
cmd:text("Options:")
cmd:option("-dir", "./data/train", 'target dir')
cmd:option("-outputdir", "./data", 'output dir')
cmd:option("-label", "", 'label file')
cmd:option("-prefix", "train_sax", 'data prefix')
cmd:option("-image_size", 64, 'image size')
cmd:option("-calibration_mergin", 16, 'calibration mergin')

opt = cmd:parse(arg)
print(opt)

local labels = nil
if opt.label and opt.label:len() > 0 then
   labels = load_labels(opt.label)
end
local x, tag, y, id  = process_data(opt.dir, labels)

torch.save(path.join(opt.outputdir, string.format("%s_x_%d_%d.t7", opt.prefix, opt.image_size, opt.calibration_mergin)), x)
torch.save(path.join(opt.outputdir, string.format("%s_tag_%d_%d.t7", opt.prefix, opt.image_size, opt.calibration_mergin)), tag)
if opt.label and opt.label:len() > 0 then
   torch.save(path.join(opt.outputdir, string.format("%s_y_%d_%d.t7", opt.prefix, opt.image_size, opt.calibration_mergin)), y)
end
torch.save(path.join(opt.outputdir, string.format("%s_id.t7", opt.prefix)), id)
