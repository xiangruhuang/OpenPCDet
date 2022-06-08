#!/bin/bash

for split in training validation; do
  for suffix in point label instance box_ladn top_lidar_origin rgb; do
    ls waymo_seg_with_r2_top_${split}.${suffix};
  done
done
