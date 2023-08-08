import torch
import pytorch_quantization
from pytorch_quantization import nn as quant_nn

print(pytorch_quantization.__version__)

import os
import numpy as np
from models.experimental import End2End



if __name__ == '__main__':
    pt_file = 'runs/train/exp/weights/best.pt'

    # 模型加载
    device = torch.device('cuda:0')
    ckpt = torch.load(pt_file, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    q_model = ckpt['model']
    q_model.eval()
    q_model = End2End(q_model).cuda().float()


    # Set static member of TensorQuantizer to use Pytorch’s own fake quantization functions
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Exporting to ONNX
    # dummy_input = torch.randn(26, 3, 384, 1280, device='cuda')
    im = np.load('im.npy') # 重要：真实图像
    dummy_input = torch.from_numpy(im).cuda()
    dummy_input = dummy_input.float()
    dummy_input = dummy_input / 255
    input_names = ["images"]
    output_names = ['num_dets', 'det_boxes']
    save_path = '/'.join(pt_file.split('/')[:-1])
    onnx_file = os.path.join(save_path, 'best_nms_dynamic_qat.onnx')
    dynamic = {'images': {0: 'batch'}}
    dynamic['num_dets'] = {0: 'batch'}
    dynamic['det_boxes'] = {0: 'batch'}
    torch.onnx.export(
        q_model,
        dummy_input,
        onnx_file,
        verbose=False,
        opset_version=13,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic)
