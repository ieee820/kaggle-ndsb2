#!/bin/sh
if [ $# -eq 1 ]; then
    seed=$1
else
    exit
fi

stdbuf -o 0 th train.lua -model 1 -seed $seed -y_sigma 0.1  -mode 1 > ./$seed-1-1.log 2>&1
stdbuf -o 0 th train.lua -model 1 -seed $seed -y_sigma 0.03 -mode 2 > ./$seed-1-2.log 2>&1
stdbuf -o 0 th train.lua -model 2 -seed $seed -y_sigma 0.1  -mode 1 > ./$seed-2-1.log 2>&1
stdbuf -o 0 th train.lua -model 2 -seed $seed -y_sigma 0.03 -mode 2 > ./$seed-2-2.log 2>&1

