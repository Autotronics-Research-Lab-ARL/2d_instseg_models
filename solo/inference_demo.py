from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import pathlib
import argparse
import mmcv


default_config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
light_config_file = '../configs/solov2/solov2_light_512_dcn_r50_fpn_8gpu_3x.py'
heavy_config_file = '../configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
default_ckpt_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'
light_ckpt_file = '../checkpoints/SOLOv2_LIGHT_512_DCN_R50_3x.pth'
heavy_ckpt_file = '../checkpoints/SOLOv2_X101_DCN_3x.pth'

# build the model from a config file and a checkpoint file


# test a single image
# img = 'demo.jpg'
parser = argparse.ArgumentParser(prog='inference_demo')
parser.add_argument('--src-dir', dest='src_dir', type=str, required=True)
parser.add_argument('--dst-dir', dest='dst_dir', type=str, required=True)
parser.add_argument('--ext', dest='ext', type=str, required=True)
parser.add_argument('--model', dest='model', type=str, required=True)


def main():
	args = parser.parse_args()
	src_dir = pathlib.Path(args.src_dir)
	dst_dir = pathlib.Path(args.dst_dir)
	dst_dir.mkdir(parents=True, exist_ok=True)
	if args.model == 'light':
		config_file = light_config_file
		checkpoint_file = light_ckpt_file
	elif args.model == 'heavy':
		config_file = heavy_config_file
		checkpoint_file = heavy_ckpt_file
	else:
		config_file = default_config_file
		checkpoint_file = default_ckpt_file

	model = init_detector(config_file, checkpoint_file, device='cuda:0')
	images = sorted(src_dir.glob('*'+args.ext))
	n_imgs = len(images)
	if n_imgs > 0:
		print('[FOUND] ---> %i' % len(images))
		for i, img in enumerate(images):
			#print('\r[FOUND] ---> %i/%i (%s)' % (i, len(images), img), end='')
			result = inference_detector(model, str(src_dir/img.name))
			print(">>>> ", type(result[0][0]), len(result[0][0]))
			print(">>>> ", type(result[0][1]), len(result[0][1]))
			print(">>>> ", type(result[0][2]), len(result[0][2]))
			break
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
