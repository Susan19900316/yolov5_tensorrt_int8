import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models, datasets

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from tqdm import tqdm

print(pytorch_quantization.__version__)

import os
import tensorrt as trt
import numpy as np
import time
import wget
import tarfile
import shutil
import cv2
import random

from models.yolo import Model
from models.experimental import End2End

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()

def collect_stats(model, data_loader):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, image in tqdm(enumerate(data_loader)):
        model(image.cuda())

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def get_crop_bbox(img, crop_size):
    """Randomly get a crop bounding box."""
    margin_h = max(img.shape[0] - crop_size[0], 0)
    margin_w = max(img.shape[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    return crop_x1, crop_y1, crop_x2, crop_y2

def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    return img

class CaliData(data.Dataset):
    def __init__(self, path, num, inputsize=[384, 1280]):
        self.img_files = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('jpg')]
        random.shuffle(self.img_files)
        self.img_files = self.img_files[:num]
        self.height = inputsize[0]
        self.width = inputsize[1]

    def __getitem__(self, index):
        f = self.img_files[index]
        img = cv2.imread(f)  # BGR
        crop_size = [self.height, self.width]
        crop_bbox = get_crop_bbox(img, crop_size)
        # crop the image
        img = crop(img, crop_bbox)
        img = img.transpose((2, 0, 1))[::-1, :, :]  # BHWC to BCHW ,BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.
        return img

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    pt_file = 'runs/train/exp/weights/best.pt'
    calib_path = 'XX/train'
    num = 2000 # 用来校正的数目
    batchsize = 4
    # 准备数据
    dataset = CaliData(calib_path, num)
    dataloader = data.DataLoader(dataset, batch_size=batchsize)

    # 模型加载
    quant_modules.initialize() #保证原始模型层替换为量化层
    device = torch.device('cuda:0')
    ckpt = torch.load(pt_file, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    # QAT
    q_model = ckpt['model']
    yaml = ckpt['model'].yaml
    q_model = Model(yaml, ch=yaml['ch'], nc=yaml['nc']).to(device)  # creat
    q_model.eval()
    q_model = End2End(q_model).cuda()
    ckpt = ckpt['model']
    modified_state_dict = {}
    for key, val in ckpt.state_dict().items():
        # Remove 'module.' from the key names
        if key.startswith('module'):
            modified_state_dict[key[7:]] = val
        else:
            modified_state_dict[key] = val
    q_model.model.load_state_dict(modified_state_dict)


    # Calibrate the model using calibration technique.
    with torch.no_grad():
        collect_stats(q_model, dataloader)
        compute_amax(q_model, method="entropy")

    # Set static member of TensorQuantizer to use Pytorch’s own fake quantization functions
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Exporting to ONNX
    dummy_input = torch.randn(26, 3, 384, 1280, device='cuda')
    input_names = ["images"]
    output_names = ["num_dets", 'det_boxes']
    # output_names = ['outputs']
    save_path = '/'.join(pt_file.split('/')[:-1])
    onnx_file = os.path.join(save_path, 'best_ptq.onnx')
    dynamic = dict()
    dynamic['num_dets'] = {0: 'batch'}
    dynamic['det_boxes'] = {0: 'batch'}
    torch.onnx.export(
        q_model,
        dummy_input,
        onnx_file,
        verbose=False,
        opset_version=13,
        do_constant_folding=False,
        output_names=output_names,
        dynamic_axes=dynamic)