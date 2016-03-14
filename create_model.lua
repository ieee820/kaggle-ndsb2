require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path

require 'cunn'

local USE_CUDNN = false
local ALPHA = 0.1
local TAG_FEAT_N = 20

if USE_CUDNN then
   require 'cudnn'
   cudnn.benchmark = true
   function cudnn.SpatialConvolution:reset(stdv)
      stdv = math.sqrt(2 / ((1.0 + ALPHA * ALPHA) * self.kW * self.kH * self.nOutputPlane))
      self.weight:normal(0, stdv)
      self.bias:zero()
   end
   function cudnn.VolumetricConvolution:reset(stdv)
      stdv = math.sqrt(2 / ((1.0 + ALPHA * ALPHA) * self.kT * self.kW * self.kH * self.nOutputPlane))
      self.weight:normal(0, stdv)
      self.bias:zero()
   end
end
function nn.SpatialConvolution:reset(stdv)
   stdv = math.sqrt(2 / ((1.0 + ALPHA * ALPHA) * self.kW * self.kH * self.nOutputPlane))
   self.weight:normal(0, stdv)
   self.bias:zero()
end
function nn.VolumetricConvolution:reset(stdv)
   stdv = math.sqrt(2 / ((1.0 + ALPHA * ALPHA) * self.kT * self.kW * self.kH * self.nOutputPlane))
   self.weight:normal(0, stdv)
   self.bias:zero()
end
function nn.Linear:reset(stdv)
   stdv = math.sqrt(2 / ((1.0 + ALPHA * ALPHA) * self.weight:size(2)))
   self.weight:normal(0, stdv)
   self.bias:zero()
end
function VolumetricConvolution(nInputPlane, nOutputPlane,
			       kT, kW, kH, dT, dW, dH, padT, padW, padH)
   if USE_CUDNN then
      return cudnn.VolumetricConvolution(nInputPlane, nOutputPlane,
				  kT, kW, kH, dT, dW, dH, padT, padW, padH)
   else
      return nn.VolumetricConvolution(nInputPlane, nOutputPlane,
				      kT, kW, kH, dT, dW, dH, padT, padW, padH)
   end
end
function VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   if USE_CUDNN then
      return cudnn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   else
      return nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   end
end
function VolumetricAveragePooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   if USE_CUDNN then
      return cudnn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   else
      return nn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH, padT, padW, padH)
   end
end

local function create_model_1max()
   local model = nn.Sequential()
   local pt = nn.ParallelTable()
   local cnn = nn.Sequential()

   -- input: Bx16x3x64x64
   cnn:add(VolumetricConvolution(16, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricMaxPooling(1, 2, 2))

   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1)) -- conv 3d
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricAveragePooling(3, 32, 32))

   cnn:add(nn.View(64))
   cnn:add(nn.Dropout(0.5))
   cnn:add(nn.Linear(64, 256))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(nn.Dropout(0.5))
   cnn:add(nn.Linear(256, 16))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(nn.Linear(16, 1))

   local tnet = nn.Sequential()
   tnet:add(nn.Linear(TAG_FEAT_N, 16))
   tnet:add(nn.ReLU())
   tnet:add(nn.Linear(16, 16))
   tnet:add(nn.ReLU())
   tnet:add(nn.Linear(16, 1))

   pt:add(cnn)
   pt:add(tnet)

   model:add(pt)
   model:add(nn.CMulTable())
   model:add(nn.View(1))

   return model
end

local function create_model_2max()
   local model = nn.Sequential()
   local pt = nn.ParallelTable()
   local cnn = nn.Sequential()

   -- input: Bx16x3x64x64
   cnn:add(VolumetricConvolution(16, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(64, 64, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricMaxPooling(1, 2, 2))

   cnn:add(VolumetricConvolution(64, 96, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(96, 96, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricMaxPooling(1, 2, 2))

   cnn:add(VolumetricConvolution(96, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 128, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(128, 256, 1, 3, 3, 1, 1, 1, 0, 1, 1))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1))   -- conv 3d
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(VolumetricAveragePooling(3, 16, 16))

   cnn:add(nn.View(256))
   cnn:add(nn.Dropout(0.5))
   cnn:add(nn.Linear(256, 512))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(nn.Dropout(0.5))
   cnn:add(nn.Linear(512, 16))
   cnn:add(nn.LeakyReLU(ALPHA))
   cnn:add(nn.Linear(16, 1))

   -- input:
   local tnet = nn.Sequential()
   tnet:add(nn.Linear(TAG_FEAT_N, 16))
   tnet:add(nn.ReLU())
   tnet:add(nn.Linear(16, 16))
   tnet:add(nn.ReLU())
   tnet:add(nn.Linear(16, 1))

   pt:add(cnn)
   pt:add(tnet)

   model:add(pt)
   model:add(nn.CMulTable())
   model:add(nn.View(1))

   return model
end
local function create_model(model_id)
   if model_id == 1 then
      return create_model_1max():cuda()
   else
      return create_model_2max():cuda()
   end
end
--model = create_model_1max()
--model:cuda()
--print(model:forward({torch.Tensor(32, 16, 3, 64, 64):uniform():cuda(), torch.Tensor(32, TAG_FEAT_N):cuda()}):size())

return create_model
