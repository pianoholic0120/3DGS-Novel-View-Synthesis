#!/bin/bash

SPLIT_FOLDER=$1
OUTPUT_FOLDER=$2

python3 ./gaussian_splatting/infer.py -m ./gaussian_splatting/output/setting_1/ -s "$SPLIT_FOLDER" --output_folder "$OUTPUT_FOLDER" --skip_test