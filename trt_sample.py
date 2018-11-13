# This sample uses a UFF MNIST model to create a TensorRT Inference Engine
from random import randint
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
  MODEL_FILE = os.path.join(os.path.dirname(__file__), "se-resnext.uff")
  INPUT_NAME ="tf_feed_image"
  INPUT_SHAPE = (3, 224, 224)
  OUTPUT_NAME = "softmax"

def build_engine(model_file):
  # For more information on TRT basics, refer to the introductory samples.
  with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    builder.max_workspace_size = common.GiB(1)
    # Parse the Uff Network
    parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
    parser.register_output(ModelData.OUTPUT_NAME)
    parser.parse(model_file, network)
    # Build and return an engine.
    return builder.build_cuda_engine(network)

# Loads a test case into the provided pagelocked_buffer.
def load_test_case(name, pagelocked_buffer):
  img = Image.open(os.path.join(name))
  # convert HWC to CHW
  planar = img.split()
  r = np.asarray(planar[0]).astype(np.float32)
  g = np.asarray(planar[1]).astype(np.float32)
  b = np.asarray(planar[2]).astype(np.float32)
  x = np.vstack((r, g, b)).flatten().astype(np.float32)
  np.copyto(pagelocked_buffer, x)
  return x

def main():
  data_path = common.find_sample_data(description="Runs a network using a UFF model file", subfolder=".")
  model_file = ModelData.MODEL_FILE

  with build_engine(model_file) as engine:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    with engine.create_execution_context() as context:
      input_tests = ["img.ppm", "ones.ppm", "orange.ppm", "panda.ppm"]
      for input_test in input_tests:
        load_test_case(input_test, pagelocked_buffer=inputs[0].host)
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(output)

if __name__ == '__main__':
  main()
  print("done")
