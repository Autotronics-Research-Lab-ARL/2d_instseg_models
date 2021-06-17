#!/bin/bash


python -W ignore tools/video_demo.py \
    --dataset coco \
    --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml \
    --ckpt data/pretrained_model/panet_mask_step179999.pth \
    --src-file data/test/kitti/videos/video.mp4 \
    --dst-file /shared_area/panet/panet_R50-FPN_output/outvid.mp4


