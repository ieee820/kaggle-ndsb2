
local cmd = torch.CmdLine()
cmd:text()
cmd:option("-a", "", 'input file c')
cmd:option("-b", "", 'input file b')
cmd:option("-o", "", 'output file')

local opt = cmd:parse(arg)

local a = torch.load(opt.a)
local b = torch.load(opt.b)

for i = 1, #b do
   table.insert(a, b[i])
end
torch.save(opt.o, a)

