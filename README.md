## Requirement ##
1. TensorRT 5.0 GA
2. Tensorflow with GPU support
3. PyCUDA
4. Python3
5. Cmake (>= 3.8)

_**Assume that the PB file is located at `<path to this project>/data` and named `se-resnext.pb`**_

## Get UFF from Tensorflow protobuf ##

```sh
$ cd <path to this project>/data
$ python3 <path to uff-converter-tf>/convert_to_uff.py <your PB file> -p preprocess.py
```

For instance:
```sh
$ cd data
$ python3 /usr/local/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py se-resnext.pb -p preprocess.py
# or
$ python3 /usr/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py se-resnext.pb -p preprocess.py
```

You should get an UFF file which may be named `se-resnext.uff` in `data` folder

## Verify TensorRT result in FP32 mod 

### Run Tensorflow ###
```sh
$ cd <path to this project>/verification
$ python3 tf_sample.py
```

### Run TensorRT ###
```sh
$ cd <path to this project>/verification
$ python3 trt_sample.py
```

The results should be same.

## Run performance benchmark ##
```sh
$ cd <path to this project>/data
$ <path to TensorRT>/trtexec --uff=<your UFF file> --output=softmax --uffInput=<input name>,3,224,224 --batch=<batch size>
```

For instance:
```sh
$ cd data
$ /usr/src/tensorrt/bin/trtexec --uff=se-resnext.uff --output=softmax --uffInput=tf_feed_image,3,224,224 --batch=32
```

## Build TensorRT engine (C++) ##
```sh
$ mkdir <path to this project>/build
$ cd <path to this project>/build
$ cmake ..
$ make -j2
```

## Run TensorRT engine ##
```sh
$ cd <path to this project>/build
# It is slow at first time because of generating TensorRT engine binary
$ ./trt_se_resnext
# The executable will use TensorRT engine binary at second time. It will be much faster in initialization
$ ./trt_se_resnext
```

## Test environment ##
1. CUDA 10
2. cuDNN 7.3.1
3. TensorRT 5.0 GA
4. Tensorflow 18.11-py3 from NGC
5. Ubuntu 16.04
6. Cmake 3.8