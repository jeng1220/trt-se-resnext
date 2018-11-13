import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
import os

def load_test_case(name):
  img = Image.open(os.path.join(name))
  # convert HWC to CHW
  planar = img.split()
  r = np.asarray(planar[0]).astype(np.float32)
  g = np.asarray(planar[1]).astype(np.float32)
  b = np.asarray(planar[2]).astype(np.float32)
  x = np.vstack((r, g, b)).flatten().astype(np.float32)
  x = x.reshape((1, 3, 224, 224))
  return x

def main():
  sess = tf.Session()
  with gfile.FastGFile('baidu_tf_se-resnext.pb', 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')

  sess.run(tf.global_variables_initializer())

  # set input/output
  input_x = sess.graph.get_tensor_by_name('tf_feed_image:0')
  op = sess.graph.get_tensor_by_name('softmax:0')

  input_tests = ["img.ppm", "ones.ppm", "orange.ppm", "panda.ppm"]
  for input_test in input_tests:
    img = load_test_case(input_test)
    [ret] = sess.run(op,  feed_dict={input_x: img})
    print(ret)

if __name__ == "__main__":
  main()
  print("done")
