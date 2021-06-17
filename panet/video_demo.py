from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange
import pathlib

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument('--dataset', required=True, dest='dataset', help='training dataset')
    parser.add_argument('--cfg', dest='cfg', required=True, help='config file')
    parser.add_argument('--ckpt', dest='ckpt', help='path of checkpoint to load')
    parser.add_argument('--src-file', dest='vidin', required=True, help='source video file for demo')
    parser.add_argument('--dst-file', dest='vidout', help='resulted video file', default="data/test/outvid.mp4")
    parser.add_argument('--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')
    args = parser.parse_args()
    return args


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4) 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in R$
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image



def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg))
    cfg_from_file(args.cfg)


    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.ckpt:
        load_name = args.ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()
    maskRCNN.eval()
    vidin_path = pathlib.Path(args.vidin)
    vidout_path = pathlib.Path(args.vidout)
    video_writer = None
    if not vidout_path.parent.exists():
        vidout_path.parent.mkdir(parents=True, exist_ok=True)
    video_reader = cv2.VideoCapture(str(vidin_path))
    src_fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    n_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    src_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.vidout:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.vidout, fourcc, src_fps,
            (src_width, src_height)
        )
    i = 0
    while True:
        i += 1
        ret_succ, frame = video_reader.read()
        if not ret_succ: break
        else:
            print("\r[LOGGING] frame ({%i}/{%i})" % (i, n_frames))
            timers = defaultdict(Timer)
            cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, frame, timers=timers)
            frame = vis_utils.vis_one_image(
                frame[:, :, ::-1],  # BGR -> RGB for visualization
                'none',
                args.vidout,
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
            )
            frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            if args.vidout:
                video_writer.write(frame)
    if video_writer:
        video_writer.release()



if __name__ == '__main__':
    main()
