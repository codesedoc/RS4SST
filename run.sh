#!/bin/bash
set -e

source docker/.env
echo "Welcome to use Reduction-Synthesis!"
python  main.py --model llama2:text \
    --host $SERVER_HOST\
    --port $OLLAMA_SERVER_PORT \
    --e_host $SERVER_HOST\
    --e_port $EVALUATOR_SERVER_PORT \
    --max_batch_size 4 \
    --dataset_dir dataset/yelp_clean \
    --transfer_mode pos2neg \
    --approach reduction_synthesis