#!/bin/bash

for dist_t in 30 50 70 100; do
  python pcdet/datasets/waymo/waymo_eval_ii.py --pred_infos $1 --gt_infos data/waymo/waymo_processed_data_v0_5_0_infos_val_with_ii.pkl --sampled_interval 1 --dist_thresh ${dist_t};
done
