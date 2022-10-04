import argparse
import os

import numpy as np
from os.path import expanduser
import time

parser = argparse.ArgumentParser(description='test ai execute performance')
parser.add_argument('--w', type=str, help="weight need to test")
parser.add_argument('--m', default="trt", type=str, help="mode run ai: trt, onnx, torch")
parser.add_argument('--s', type=int, default=320, help='size input')
args = parser.parse_args()

devicename = None
try:
    import torch

    devicename = torch.cuda.get_device_name()
except:
    print("skip torch")

if __name__ == "__main__":
    s = args.s
    if args.m == "onnx":
        try:
            import onnxruntime
        except ModuleNotFoundError:
            print("run: pip install onnxruntime-gpu")

        for path in os.listdir("weight_onnx"):
            if "mobi" in path:
                s = 224
            full_path = os.path.join("weight_onnx", path)
            session = onnxruntime.InferenceSession(full_path, providers=['CUDAExecutionProvider'])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            list_time = []
            for i in range(20):
                if "yolo" in path:
                    dummy_input = np.random.rand(1, 416, 416, 3).astype(np.float32)
                else:
                    dummy_input = np.random.rand(1, 3, s, s).astype(np.float32)

                start = time.time()
                result = session.run([output_name], {input_name: dummy_input})
                list_time.append(time.time() - start)

            print(f"\nweight: {path}\nonnx runtime: {np.mean(list_time[1:]):.3f}s/sample")

    elif args.m == "trt":
        import tensorrt as trt
        import common

        if args.w:
            path = expanduser(args.w)

        logger = trt.Logger(trt.Logger.WARNING)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        list_time = []

        for i in range(20):
            dummy_input = np.random.rand(1, 3, s, s).astype(np.float32)
            start = time.time()
            img = np.ravel(dummy_input)
            np.copyto(inputs[0].host, img)
            trt_result = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            list_time.append(time.time() - start)

        print(f"\nweight: {path}\ntensorrt runtime: {np.mean(list_time[1:]):.3f}s/sample")

    elif args.m == "torch":
        import segmentation_models_pytorch as smp
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        list_time = []
        encoders = ["mobilenet_v2", "resnet34", "resnet50"]
        for enc in encoders:
            model = smp.Unet(encoder_name=enc)
            model = model.to(device)
            for i in range(20):
                dummy_input = torch.rand((1, 3, s, s), device=device)
                start = time.time()
                model(dummy_input)
                list_time.append(time.time() - start)

            print(f"\nunet encoder: {enc}\npytorch runtime: {np.mean(list_time[1:]):.3f}s/sample")

    else:
        print("Error: --m trt or onnx or torch")

    if devicename:
        print(devicename)
