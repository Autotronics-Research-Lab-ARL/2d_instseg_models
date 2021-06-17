# Instance Segmentation

## models
- [PANet](https://github.com/ShuLiu1993/PANet)
- [SOLOv2](https://github.com/WXinlong/SOLO)
- [SWIN-T](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)

## Results
<table>
    <tr>
        <th>PANet</th>
        <td><img src='assets/panet_mask_18.gif'></td>
    </tr>
    <tr>
        <th>PANet</th>
        <td><img src='assets/solo_heavy.gif'></td>
    </tr>
    <tr>
        <th>PANet</th>
        <td><img src='assets/swin_tiny.gif'></td>
    </tr>
</table>


<br><br><br>
## Installation
select the model to
``` bash
docker build -t <image-name> <Dockerfile path>
docker run --gpus all --name <container-name> -dit --ipc host \
    -v <path-on-your-pc>:/shared_area <image-name>
docker exec -it <container-name> bash
cp /shared_area/infer_model.bash /workspace/<model-name>
cp /shared_area/video_demo.py <model-repo>/tools
ln -s /shared_area/<pretrained-weights> <model-repo>/checkpoints
```
<br>

model   |   config  |   checkpoint
:------:|:---------:|:------------:
**PANet** | [e2e_panet_R-50-FPN_2x_mask](https://github.com/ShuLiu1993/PANet/blob/master/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml) | [panet_mask_step179999](https://drive.google.com/u/0/uc?id=1-pVZQ3GR6Aj7KJzH9nWoRQ-Lts8IcdMS&export=download)
**SOLOv2** | [solov2_x101_dcn_fpn_8gpu_3x](https://github.com/WXinlong/SOLO/blob/master/configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py) | [SOLOv2_X101_DCN_3x](https://cloudstor.aarnet.edu.au/plus/s/KV9PevGeV8r4Tzj/download)
**SWIN-T** | [mask_rcnn_swin_tiny_patch4_3x_coco](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [mask_rcnn_swin_tiny_patch4_window7](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.pth)
