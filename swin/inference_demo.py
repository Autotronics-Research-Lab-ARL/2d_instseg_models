from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import pathlib
import argparse
import mmcv


config_file = '../configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/mask_rcnn_swin_small_patch4_window7.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
# img = 'demo.jpg'
parser = argparse.ArgumentParser(prog='inference_demo')
parser.add_argument('--src-dir', dest='src_dir', type=str, required=True)
parser.add_argument('--dst-dir', dest='dst_dir', type=str, required=True)
parser.add_argument('--ext', dest='ext', type=str, required=True)


def main():
	args = parser.parse_args()
	src_dir = pathlib.Path(args.src_dir)
	dst_dir = pathlib.Path(args.dst_dir)
	dst_dir.mkdir(parents=True, exist_ok=True)
	images = sorted(src_dir.glob('*'+args.ext))
	n_imgs = len(images)
	if n_imgs > 0:
		print('[FOUND] ---> %i' % len(images))
		for i, img in enumerate(images):
			print('\r[FOUND] ---> %i/%i (%s)' % (i, len(images), img), end='')
			result = inference_detector(model, str(src_dir/img.name))
			show_result_ins(
				str(src_dir/img.name), 
				result, 
				model.CLASSES, 
				score_thr=0.25, 
				out_file=str(dst_dir/img.name)
			)
		print('\n', '-'*50, '\n', 'processes all images')

if __name__ == "__main__":
	main()
