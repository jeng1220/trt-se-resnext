#include "common.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg)
{
  // suppress info-level messages
  if (severity == Severity::kINFO) return;
  switch (severity)
  {
    case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
    case Severity::kERROR:          std::cerr << "ERROR: ";          break;
    case Severity::kWARNING:        std::cerr << "WARNING: ";        break;
    case Severity::kINFO:           std::cerr << "INFO: ";           break;
    default:                        std::cerr << "UNKNOWN: ";        break;
  }
  std::cerr << msg << std::endl;
}

void InplaceConvertRGBtoBGR(std::vector<uint8_t>& src)
{
  auto count = src.size();
  assert(count % 3 == 0);
  for (size_t i = 0; i < count; i += 3)
  {
    auto tmp = src[i];
    size_t idx = i + 2;
    // swap
    src[i] = src[idx];
    src[idx] = tmp;
  }
}

std::vector<uint8_t> ConvertHWCtoCHW(const std::vector<uint8_t>& src,
  int batch, int h, int w, int channel)
{
  auto count = src.size();
  assert(count == static_cast<size_t>(batch) * h * w * channel);
  std::vector<uint8_t> dst(count);

  const uint8_t* src_ptr = src.data();

  for (int n = 0; n < batch; ++n)
  {
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        for (int c = 0; c < channel; ++c)
        {
          size_t idx = n * (h*w*channel)
            + c * (h*w)
            + y * (w)
            + x;
          dst[idx] = *src_ptr;
          src_ptr++;
        }
      }
    }
  }
  return dst;
}

std::vector<uint8_t> ReadPPMFile(const std::string& fn,
  bool convert_bgr, bool convert_plane)
{
  std::ifstream fs;
  fs.open(fn, std::ifstream::in | std::ifstream::binary);
  assert(fs.is_open());
  std::string magic, h, w, max;
  fs >> magic;
  assert(magic.compare("P6") == 0);
  fs >> h;
  assert(h.compare("224")    == 0);
  fs >> w;
  assert(w.compare("224")    == 0);
  fs >> max;
  assert(max.compare("255")  == 0);
  auto i32h = std::stoi(h);
  auto i32w = std::stoi(w);
  size_t i8count = 3 * i32h * i32w;
  std::vector<uint8_t> i8buff(i8count);
  fs.seekg(1, fs.cur);
  fs.read(reinterpret_cast<char*>(i8buff.data()), i8count);

  if (convert_bgr) {
    InplaceConvertRGBtoBGR(i8buff);
  }

  if (convert_plane) {
    return ConvertHWCtoCHW(i8buff,
      1, i32h, i32w, 3);
  }

  fs.close();
  return i8buff;
}

std::vector<float> ImagePreprocess(const std::vector<uint8_t>& src)
{
  auto count = src.size();
  std::vector<float> dst(count);
  for (size_t i = 0; i < count; ++i)
  {
    dst[i] = static_cast<float>(src[i]);
  }
  return dst;
}

std::vector<uint8_t> ReadPreBuiltInferEngine(const std::string& fn)
{
  std::vector<uint8_t> out;

  std::ifstream fs;
  fs.open(fn, std::ifstream::in | std::ifstream::binary);
  if (fs.is_open() == false) return out;

  // get length of file:
  fs.seekg(0, fs.end);
  int length = fs.tellg();
  fs.seekg(0, fs.beg);
  // allocate memory
  out.resize(length);
  // read data as a block
  fs.read(reinterpret_cast<char*>(out.data()),length);
  fs.close();
  return out;
}