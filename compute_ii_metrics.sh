#!/bin/bash

mkdir -p $2
for dist_t in 30 70; do
  python pcdet/datasets/waymo/waymo_eval_ii.py --pred_infos $1 --gt_infos data/waymo/waymo_processed_data_v0_5_0_infos_val_with_ii.pkl --sampled_interval 1 --dist_thresh ${dist_t} --output_path $2;
  python pcdet/datasets/waymo/waymo_eval_ii.py --pred_infos $1 --gt_infos data/waymo/waymo_processed_data_v0_5_0_infos_val_with_ii.pkl --sampled_interval 1 --dist_thresh ${dist_t} --l1 --output_path $2;
done
