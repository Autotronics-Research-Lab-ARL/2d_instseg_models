#!/bin/bash

Help() {
    echo
    echo usage: [-h|--i]
    echo options:
    echo i	input video path
    echo
}

while getopts "i:h" opt
do
    case "$opt" in
    i )	   VIDEO_PATH="$OPTARG";;
    h )    Help;;
    ? )    Help;;
    esac
done

python demo/video_demo.py \
    "$VIDEO_PATH" \
    configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py \
    checkpoints/SOLOv2_X101_DCN_3x.pth \
    --out /shared_area/inst_seg/solo/heavy_output/outvid.mp4 
