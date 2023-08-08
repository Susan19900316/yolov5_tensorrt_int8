import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2


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

class yolov5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, imgpath, batch_size, channel, inputsize=[384, 1280]):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = 'yolov5.cache'
        self.batch_size = batch_size
        self.Channel = channel
        self.height = inputsize[0]
        self.width = inputsize[1]
        self.imgs = [os.path.join(imgpath, file) for file in os.listdir(imgpath) if file.endswith('jpg')]
        np.random.shuffle(self.imgs)
        self.imgs = self.imgs[:2000]
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
        # self.data_size = trt.volume([self.batch_size, self.Channel, self.height, self.width]) * trt.float32.itemsize
        self.data_size = self.calibration_data.nbytes
        self.device_input = cuda.mem_alloc(self.data_size)
        # self.device_input = cuda.mem_alloc(self.calibration_data.nbytes)

    def free(self):
        self.device_input.free()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs)
            return [self.device_input]
        except:
            print('wrong')
            return None
    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.height, self.width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = cv2.imread(f)  # BGR
                crop_size = [self.height, self.width]
                crop_bbox = get_crop_bbox(img, crop_size)
                # crop the image
                img = crop(img, crop_bbox)
                img = img.transpose((2, 0, 1))[::-1, :, :]  # BHWC to BCHW ,BGR to RGB
                img = np.ascontiguousarray(img)
                img = img.astype(np.float32) / 255.
                assert (img.nbytes == self.data_size / self.batch_size), 'not valid img!' + f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            # os.fsync(f)


def get_engine(onnx_file_path, engine_file_path, cali_img, mode='FP32', workspace_size=4096):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    def build_engine():
        assert mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            explicit_batch_flag
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            config.max_workspace_size = workspace_size * (1024 * 1024)  # workspace_sizeMiB
            # 构建精度
            if mode.lower() == 'fp16':
                config.flags |= 1 << int(trt.BuilderFlag.FP16)

            if mode.lower() == 'int8':
                print('trt.DataType.INT8')
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                config.flags |= 1 << int(trt.BuilderFlag.FP16)
                calibrator = yolov5EntropyCalibrator(cali_img, 26, 3, [384, 1280])
                # config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
                config.int8_calibrator = calibrator
            # if True:
            #     config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, min=(1, 3, 384, 1280), opt=(12, 3, 384, 1280), max=(26, 3, 384, 1280))
            config.add_optimization_profile(profile)
            # config.set_calibration_profile(profile)
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            # plan = builder.build_serialized_network(network, config)
            # engine = runtime.deserialize_cuda_engine(plan)
            engine = builder.build_engine(network,config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                # f.write(plan)
                f.write(engine.serialize())
            return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main(onnx_file_path, engine_file_path, cali_img_path, mode='FP32'):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    get_engine(onnx_file_path, engine_file_path, cali_img_path, mode)


if __name__ == "__main__":
    onnx_file_path = '/home/models/boatdetect_yolov5/last_nms_dynamic.onnx'
    engine_file_path = "/home/models/boatdetect_yolov5/last_nms_dynamic_onnx2trtptq.plan"
    cali_img_path = '/home/data/frontview/test'
    main(onnx_file_path, engine_file_path, cali_img_path, mode='int8')
