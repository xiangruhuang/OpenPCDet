#!/bin/bash

for dist_t in 30 50 70 100; do
  python pcdet/datasets/waymo/waymo_eval_ii.py --pred_infos output/waymo_models/pv_rcnn_plusplus_resnet_gtaug/default/eval/eval_with_train/epoch_30/val/result.pkl --gt_infos data/waymo/waymo_processed_data_v0_5_0_infos_val_with_ii.pkl --sampled_interval 1 --dist_thresh ${dist_t};
done
