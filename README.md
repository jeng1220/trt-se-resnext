## requirement ##
1. TensorRT 5.0 GA
2. Tensorflow with GPU support
3. PyCUDA
4. Python3

## get UFF from Tensorflow protobuf ##
```sh
python3 <path to uff-converter-tf>/convert_to_uff.py <your PB file> -p preprocess.py
```

for instance:
```sh
python3 /usr/local/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py se-resnext.pb -p preprocess.py
# or
python3 /usr/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py se-resnext.pb -p preprocess.py
```

## to verify TensorRT result in FP32 mod 

### run Tensorflow ###
```sh
$ python3 tf_sample.py
```

### run TensorRT ###
```sh
$ python3 trt_sample.py
```

the results should be same.

## to run performance benchmark ##
```sh
<path to TensorRT>/trtexec --uff=<your UFF file> --output=softmax --uffInput=<input name>,3,224,224 --batch=<batch size>
```

for instance:
```sh
/usr/src/tensorrt/bin/trtexec --uff=se-resnext.uff --output=softmax --uffInput=tf_feed_image,3,224,224 --batch=32
```

## test environment ##
1. CUDA 10
2. cuDNN 7.3.1
3. TensorRT 5.0 GA
4. Tensorflow 18.11-py3 from NGC
5. Ubuntu 16.04
