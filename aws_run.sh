#!/bin/bash

key_file=$HOME/.ssh/calfornia.pem

declare -a ips=(
# ...
)

seed=71
for ip in ${ips[@]}
do
    echo "******* ${ip}"
    ssh -i ${key_file} ubuntu@${ip} "source /home/ubuntu/.zshrc; cd ndsb2/sax ; nohup ./train.sh ${seed} </dev/null >./run.log 2>&1 &"
    seed=`expr ${seed} + 1`
done
