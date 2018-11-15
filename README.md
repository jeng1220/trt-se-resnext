## requirement ##
1. TensorRT 5.0 GA
2. Tensorflow with GPU support
3. PyCUDA
4. Python3

assume that the PB file is located at `<path to this project>/data` and named `se-resnext.pb`

## get UFF from Tensorflow protobuf ##

```sh
cd <path to this project>/data
python3 <path to uff-converter-tf>/convert_to_uff.py <your PB file> -p preprocess.py
```

for instance:
```sh
cd data
python3 /usr/local/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py se-resnext.pb -p preprocess.py
# or
python3 /usr/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py se-resnext.pb -p preprocess.py
```

then, you should get an UFF file which may be named `se-resnext.uff` in `data` folder

## to verify TensorRT result in FP32 mod 

### run Tensorflow ###
```sh
cd <path to this project>/verification
$ python3 tf_sample.py
```

### run TensorRT ###
```sh
cd <path to this project>/verification
$ python3 trt_sample.py
```

the results should be same.

## to run performance benchmark ##
```sh
cd <path to this project>/data
<path to TensorRT>/trtexec --uff=<your UFF file> --output=softmax --uffInput=<input name>,3,224,224 --batch=<batch size>
```

for instance:
```sh
cd data
/usr/src/tensorrt/bin/trtexec --uff=se-resnext.uff --output=softmax --uffInput=tf_feed_image,3,224,224 --batch=32
```

## test environment ##
1. CUDA 10
2. cuDNN 7.3.1
3. TensorRT 5.0 GA
4. Tensorflow 18.11-py3 from NGC
5. Ubuntu 16.04
