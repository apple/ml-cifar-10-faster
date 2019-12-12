#! /bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#

# Starts WORKERS python processes with sequential values for the RANK and LOCAL_RANK environment variables.
# Forwards any command line arguments to the child processes

function run_worker {
    MASTER_ADDR=localhost MASTER_PORT=123456 WORLD_SIZE=$1 RANK=$2 LOCAL_RANK=$2 python3.6 fast_cifar_10_distributed.py ${@:3}
}

echo "CIFAR 10 training run with ${WORKERS} workers"

# Run all but the last worker in the background
for i in `seq 0 $((WORKERS-2))`
do
    run_worker $WORKERS $i ${@:1} &
done

# Run the last process in the foreground
run_worker $WORKERS $((WORKERS-1)) ${@:1}
