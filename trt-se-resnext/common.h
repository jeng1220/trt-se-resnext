#pragma once

#include "NvInfer.h"
#include <cstdint>
#include <string>
#include <vector>

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger
{
  public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override;
};

std::vector<uint8_t> ReadPPMFile(const std::string& fn,
  bool convert_bgr = true, bool convert_plane = true);

std::vector<float> ImagePreprocess(const std::vector<uint8_t>& src);

std::vector<uint8_t> ReadPreBuiltInferEngine(const std::string& fn);