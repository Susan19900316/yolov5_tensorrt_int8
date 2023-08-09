## 介绍

- 基于tensorrt官方文档，基于yolov5_v7.0结合自身训练模型对yolov5进行int8量化；
- 关于tensorrt int8量化的详细文档请参考本人[CSDN](https://blog.csdn.net/Suan2014/article/details/132168130)和[知乎相关文章](https://zhuanlan.zhihu.com/p/648877516);

## 准备工作

- 下载[yolov5](https://github.com/ultralytics/yolov5)官方代码，本人实验是基于yolov5_v7.0版本进行的；
- 按照官方教程准备数据集，对yolov5模型进行训练，获得要部署的模型best.pt或者last.pt，后文默认采用best.pt；
- 将best.pt导出为onnx;
- 将本仓的代码放置于yolov5内，与train.py同级；

## QuickStart

### 方式1：PTQ(训练后量化)

PTQ共有3种实现方式，示例如下：

#### 方式1.1 engine序列化前进行int8量化

```shell
python onnx2trt_ptq.py
```

**参数说明：**

- onnx_file_path：准备工作中导出的yolov5 onnx模型；
- engine_file_path：待保存的int8量化的序列化tensorrt模型路径；
- cali_img_path：校正数据集路径，本人实验时校正数据集采用了2000张训练数据集子集；

**注意事项：**

- 实验的是输入为动态batch的情况，故添加下述代码，请根据模型情况自行调整输入

```python
 profile = builder.create_optimization_profile()
 profile.set_shape(network.get_input(0).name, min=(1, 3, 384, 1280), opt=(12, 3, 384, 1280), max=(26, 3, 384, 1280))
 config.add_optimization_profile(profile)
```

- 校正数据集的处理请根据自身模型输入自行调整；
- **校正数据集的shape[B, C,H,W]需与推理时模型输入shape[B,C,H,W]完全一致**，方能得到较好的量化结果，即该量化方式不适用于动态shape;

#### 方式1.2 **polygraphy工具**:应该是对1.1量化过程的封装

- 安装polygraphy

  ```shell
  pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
  ```

- 量化

```shell
polygraphy convert XX.onnx --int8 --fp16 --data-loader-script loader_data.py --calibration-cache XX.cache -o XX.trt --trt-min-shapes images:[1,3,384,1280] --trt-opt-shapes images:[26,3,384,1280] --trt-max-shapes images:[26,3,384,1280] #量化
```

 **参数说明：**

- XX.onnx：准备工作中导出的yolov5 onnx模型；
- XX.trt: 待保存的int8量化的序列化tensorrt模型路径；
- XX.cache: 保存的校正cache；
- loader_data.py: 量化时，会读取loader_data.py中的load_data函数；

**注意事项：**

- loader_data.py中只是示例代码，请将此代码换做加载校正数据集，并按照模型需求处理校正数据集；
- 校正数据集的shape要与上述命令中的--trt-opt-shapes一致，否则报错；
- **校正数据集的shape[B, C,H,W]需与推理时模型输入shape[B,C,H,W]完全一致**，方能得到较好的量化结果，即该量化方式不适用于动态shape;

#### 方式1.3 pytorch中执行(推荐)

```shell
python pytorch_yolov5_ptq.py
```

**参数说明：**

- pt_file：准备工作中训练好的pytorch模型；
- calib_path：校正数据集路径；
- num： 校正数据集中采用num个数据进行校正；
- batchsize: 校正数据集输入的batchsize，此种方式中，**校正数据集的batchsize无需与推理时保持一致，也可得到较好的量化结果**；
- 按照自己模型的数据处理方式写CaliData类；
- 此模型导出int8量化后的onnx模型；
- 利用trtexec将onnx模型转为对应的tensorrt模型，命令中记得加入 --int8 --fp16；

## 方式2 QAT（训练感知量化）

#### 训练

```shell
python pytorch_yolov5_qat.py -m best.pt ...#best.pt为准备工作训练好的模型，...为其他命令参考train.py训练，数据采用训练数据集
```

**注意事项：**

- 训练时图像尺寸必须与推理时保持一致，方能得到较好的量化结果；
- 初始学习率lr0设为0.0001

#### pytorch导出为onnx

```shell
python export_onnx_qat.py
```

**参数说明：**

- pt_file：为上述QAT训练的模型

**注意事项：**

- 实验中，导出onnx时，输入必须**真实的图像数据**才能导出正常输出的onnx模型
