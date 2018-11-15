#include "common.h"
#include "resnext.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

int main(int argc, char** argv)
{
  std::string uff_fn       = "../data/se-resnext.uff";
  std::string input_layer  = "tf_feed_image";
  std::string output_layer = "softmax";
  std::string engine_path  = "trt_engine.bin";
  int input_channel = 3;
  int input_width   = 224;
  int input_height  = 224;
  bool convert_to_bgr   = false;
  bool enable_nhwc      = false;
  bool enable_fp16      = false;

  // initial TensorRT
  int max_batch = 1;
  ResNext* net_ptr = nullptr;

  auto engine_bin = ReadPreBuiltInferEngine(engine_path);
  if (engine_bin.size()) {
    std::cout << "detect prebuilt engine binary \"" << engine_path << "\" and use it" << std::endl;
    net_ptr = new ResNext(engine_bin.data(), engine_bin.size(), max_batch);
  }
  else {
    net_ptr = new ResNext(uff_fn,
      input_layer, output_layer,
      max_batch, input_channel, input_width, input_height,
      enable_fp16, enable_nhwc);
    
    net_ptr->save(engine_path);
  }

  // inference (forward propagation)
  std::string img_paths[4] = {
    "../data/img.ppm",
    "../data/ones.ppm",
    "../data/orange.ppm",
    "../data/panda.ppm"
  };

  for (auto& fn : img_paths) 
  {
    // read image
    auto img_buff = ReadPPMFile(fn, convert_to_bgr,
      !enable_nhwc);
    auto src_buff = ImagePreprocess(img_buff);

    // copy input from host buffer to device buffer
    int run_batch = 1;
    net_ptr->set_src_buffer(src_buff, run_batch);

    net_ptr->inference(run_batch);

    // host buffer allocation
    auto dst_count = run_batch * net_ptr->get_dst_count();
    std::vector<float> dst_buff(dst_count);

    // copy result from device buffer to host buffer
    net_ptr->get_dst_buffer(dst_buff, run_batch);

    // show result
    for (auto& y : dst_buff) 
    {
      std::cout << +y << ", ";
    }
    std::cout << std::endl;
  }

  delete net_ptr;
  return EXIT_SUCCESS;
}
