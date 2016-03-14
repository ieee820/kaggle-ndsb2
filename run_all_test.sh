#!/bin/sh

./convert_data_test.sh

# NOTICE: I used 8 g2.xlarge instances to execute following 8 lines.
# See ./aws_run.sh
./train.sh 71
./train.sh 72
./train.sh 73
./train.sh 74
./train.sh 75
./train.sh 76
./train.sh 77
./train.sh 78

./test.sh
th find_normal_param.lua
th predict.lua
