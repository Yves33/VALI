/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
 * Copyright 2021 Videonetics Technology Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyNvDecoder::PyNvDecoder(const string& pathToFile, int gpu_id)
    : PyNvDecoder(pathToFile, gpu_id, map<string, string>()) {}

PyNvDecoder::PyNvDecoder(const string& pathToFile, CUcontext ctx, CUstream str)
    : PyNvDecoder(pathToFile, ctx, str, map<string, string>()) {}

PyNvDecoder::PyNvDecoder(const string& pathToFile, int gpu_id,
                         const map<string, string>& ffmpeg_options) {
  if (gpu_id < 0 || gpu_id >= CudaResMgr::Instance().GetNumGpus()) {
    gpu_id = 0U;
  }
  gpuID = gpu_id;

  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);
  format = params.videoContext.format;

  upDecoder.reset(NvdecDecodeFrame::Make(
      CudaResMgr::Instance().GetStream(gpuID),
      CudaResMgr::Instance().GetCtx(gpuID), params.videoContext.codec,
      poolFrameSize, params.videoContext.width, params.videoContext.height,
      format));
}

PyNvDecoder::PyNvDecoder(const string& pathToFile, CUcontext ctx, CUstream str,
                         const map<string, string>& ffmpeg_options) {
  vector<const char*> options;
  for (auto& pair : ffmpeg_options) {
    options.push_back(pair.first.c_str());
    options.push_back(pair.second.c_str());
  }
  upDemuxer.reset(
      DemuxFrame::Make(pathToFile.c_str(), options.data(), options.size()));

  MuxingParams params;
  upDemuxer->GetParams(params);
  format = params.videoContext.format;

  upDecoder.reset(NvdecDecodeFrame::Make(
      str, ctx, params.videoContext.codec, poolFrameSize,
      params.videoContext.width, params.videoContext.height, format));
}

PyNvDecoder::PyNvDecoder(uint32_t width, uint32_t height,
                         Pixel_Format new_format, cudaVideoCodec codec,
                         uint32_t gpu_id)
    : format(new_format) {
  if (gpu_id >= CudaResMgr::Instance().GetNumGpus()) {
    gpu_id = 0U;
  }
  gpuID = gpu_id;

  upDecoder.reset(
      NvdecDecodeFrame::Make(CudaResMgr::Instance().GetStream(gpuID),
                             CudaResMgr::Instance().GetCtx(gpuID), codec,
                             poolFrameSize, width, height, format));
}

PyNvDecoder::PyNvDecoder(uint32_t width, uint32_t height,
                         Pixel_Format new_format, cudaVideoCodec codec,
                         CUcontext ctx, CUstream str)
    : format(new_format) {
  upDecoder.reset(NvdecDecodeFrame::Make(str, ctx, codec, poolFrameSize, width,
                                         height, format));
}

Buffer* PyNvDecoder::getElementaryVideo(SeekContext* seek_ctx,
                                        TaskExecDetails& details,
                                        bool needSEI) {
  shared_ptr<Buffer> pSeekCtxBuf = nullptr;
  do {
    // Set 1st demuxer input to any non-zero value if we need SEI;
    if (needSEI) {
      upDemuxer->SetInput((Token*)0xdeadbeefull, 0U);
    }

    // Set 2nd demuxer input to seek context if we need to seek;
    if (seek_ctx && seek_ctx->use_seek) {
      pSeekCtxBuf.reset(Buffer::MakeOwnMem(sizeof(SeekContext), seek_ctx));
      upDemuxer->SetInput((Token*)pSeekCtxBuf.get(), 1U);
    }

    if (TASK_EXEC_FAIL == upDemuxer->Execute()) {
      upDemuxer->GetExecDetails(details);
      return nullptr;
    }

    /* Clear inputs and set down seek flag or we will seek
     * for one and the same frame multiple times. */
    if (seek_ctx) {
      seek_ctx->use_seek = false;
    }
    upDemuxer->ClearInputs();
  } while (!upDemuxer->GetOutput(0U));

  auto pktDataBuf = (Buffer*)upDemuxer->GetOutput(3U);
  if (pktDataBuf) {
    auto pPktData = pktDataBuf->GetDataAs<PacketData>();
    if (seek_ctx) {
      seek_ctx->out_frame_pts = pPktData->pts;
      seek_ctx->out_frame_duration = pPktData->duration;
    }
  }

  return (Buffer*)upDemuxer->GetOutput(0U);
};

Surface* PyNvDecoder::getDecodedSurface(SeekContext* seek_ctx,
                                        TaskExecDetails& details,
                                        bool needSEI) {
  upDecoder->ClearInputs();
  upDecoder->ClearOutputs();

  Surface* surface = nullptr;
  do {
    auto elementaryVideo = getElementaryVideo(seek_ctx, details, needSEI);
    auto pktData = (Buffer*)upDemuxer->GetOutput(3U);

    upDecoder->SetInput(elementaryVideo, 0U);
    upDecoder->SetInput(pktData, 1U);

    auto ret = upDecoder->Execute();
    upDecoder->GetExecDetails(details);

    if (TASK_EXEC_FAIL == ret) {
      break;
    }

    surface = (Surface*)upDecoder->GetOutput(0U);
  } while (!surface);

  return surface;
};

Surface* PyNvDecoder::getDecodedSurfaceFromPacket(
    const py::array_t<uint8_t>* pPacket, TaskExecDetails& details,
    const PacketData* p_packet_data, bool no_eos) {
  upDecoder->ClearInputs();
  upDecoder->ClearOutputs();

  Surface* surface = nullptr;
  unique_ptr<Buffer> packetData = nullptr;
  unique_ptr<Buffer> elementaryVideo = nullptr;

  if (pPacket && pPacket->size()) {
    elementaryVideo = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(pPacket->size(), pPacket->data()));
  }

  if (no_eos) {
    upDecoder->SetInput((Token*)0xbaddf00dull, 2U);
  }

  if (p_packet_data) {
    packetData = unique_ptr<Buffer>(
        Buffer::MakeOwnMem(sizeof(PacketData), p_packet_data));
    upDecoder->SetInput(packetData.get(), 1U);
  }

  upDecoder->SetInput(elementaryVideo ? elementaryVideo.get() : nullptr, 0U);
  if (TASK_EXEC_FAIL == upDecoder->Execute()) {
    upDecoder->GetExecDetails(details);
    return nullptr;
  }

  return (Surface*)upDecoder->GetOutput(0U);
};

uint32_t PyNvDecoder::Width() const {
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.width;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get width from demuxer instead");
  }
}

void PyNvDecoder::LastPacketData(PacketData& packetData) const {
  if (upDemuxer) {
    auto mp_buffer = (Buffer*)upDemuxer->GetOutput(3U);
    if (mp_buffer) {
      auto mp = mp_buffer->GetDataAs<PacketData>();
      packetData = *mp;
    }
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get packet data from demuxer instead");
  }
}

ColorSpace PyNvDecoder::GetColorSpace() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_space;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color space from demuxer instead");
  }
}

ColorRange PyNvDecoder::GetColorRange() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.color_range;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get color range from demuxer instead");
  }
}

uint32_t PyNvDecoder::Height() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.height;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get height from demuxer instead");
  }
}

double PyNvDecoder::Framerate() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.frameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get framerate from demuxer instead");
  }
}

double PyNvDecoder::AvgFramerate() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.avgFrameRate;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get avg framerate from demuxer instead");
  }
}

bool PyNvDecoder::IsVFR() const {
  if (upDemuxer) {

    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.is_vfr;
  } else {
    throw runtime_error(
        "Decoder was created without built-in demuxer support. "
        "Please check variable framerate flag from demuxer instead");
  }
}

double PyNvDecoder::Timebase() const {
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.timeBase;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get time base from demuxer instead");
  }
}

uint32_t PyNvDecoder::Framesize() const {
  if (upDemuxer) {
    auto pSurface = Surface::Make(GetPixelFormat(), Width(), Height(),
                                  CudaResMgr::Instance().GetCtx(gpuID));
    if (!pSurface) {
      throw runtime_error("Failed to determine video frame size.");
    }
    uint32_t size = pSurface->HostMemSize();
    delete pSurface;
    return size;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get frame size from demuxer instead");
  }
}

uint32_t PyNvDecoder::Numframes() const {
  if (upDemuxer) {
    MuxingParams params;
    upDemuxer->GetParams(params);
    return params.videoContext.num_frames;
  } else {
    throw runtime_error("Decoder was created without built-in demuxer support. "
                        "Please get num_frames from demuxer instead");
  }
}

Pixel_Format PyNvDecoder::GetPixelFormat() const { return format; }

void PyNvDecoder::UpdateState() {
  last_h = Height();
  last_w = Width();
}

bool PyNvDecoder::IsResolutionChanged() {
  try {
    if (last_h != Height()) {
      return true;
    }

    if (last_w != Width()) {
      return true;
    }
  } catch (exception& e) {
    return false;
  }

  return false;
}

bool PyNvDecoder::DecodeSurface(DecodeContext& ctx, TaskExecDetails& details) {

  if (!upDemuxer && !ctx.IsStandalone() && !ctx.IsFlush()) {
    throw std::runtime_error(
        "Tried to call DecodeSurface on a Decoder that has been "
        "initialized "
        "without a built-in demuxer. Please use "
        "DecodeSurfaceFromPacket/DecodeFrameFromPacket instead or "
        "intialize the decoder with a demuxer when decoding from a file");
  }
  try {
    UpdateState();
  } catch (exception& e) {
    // Prevent exception throw;
  }

  bool loop_end = false;
  // If we feed decoder with Annex.B from outside we can't seek;
  bool const use_seek = ctx.IsSeek();
  bool dec_error = false, dmx_error = false;

  Surface* pRawSurf = nullptr;

  // Check seek params & flush decoder if we need to seek;
  if (use_seek) {
    MuxingParams params;
    upDemuxer->GetParams(params);

    if (PREV_KEY_FRAME != ctx.GetSeekContext()->mode) {
      throw runtime_error(
          "Decoder can only seek to closest previous key frame");
    }

    // Flush decoder without setting eos flag;
    Surface* p_surf = nullptr;
    do {
      try {
        p_surf = getDecodedSurfaceFromPacket(nullptr, details, nullptr);
      } catch (decoder_error& dec_exc) {
        dec_error = true;
        cerr << dec_exc.what() << endl;
      } catch (cuvid_parser_error& cvd_exc) {
        dmx_error = true;
        cerr << cvd_exc.what() << endl;
      }
    } while (p_surf && !p_surf->Empty());
    upDecoder->ClearOutputs();

    // Set number of decoded frames to zero before the loop;
    ctx.GetSeekContextMutable()->num_frames_decoded = 0U;
  }

  /* Decode frames in loop if seek was done.
   * Otherwise will return after 1st iteration. */
  do {
    try {
      if (ctx.IsFlush()) {
        pRawSurf = getDecodedSurfaceFromPacket(nullptr, details, nullptr);
      } else if (ctx.IsStandalone()) {
        pRawSurf = getDecodedSurfaceFromPacket(ctx.GetPacket(), details,
                                               ctx.GetInPacketData());
      } else {
        pRawSurf = getDecodedSurface(ctx.GetSeekContextMutable(), details,
                                     ctx.HasSEI());
      }

      if (!pRawSurf) {
        break;
      }
    } catch (decoder_error& dec_exc) {
      dec_error = true;
      cerr << dec_exc.what() << endl;
    } catch (cuvid_parser_error& cvd_exc) {
      dmx_error = true;
      cerr << cvd_exc.what() << endl;
    }

    // Increase the counter;
    if (use_seek)
      ctx.GetSeekContextMutable()->num_frames_decoded++;

    /* Get timestamp from decoder.
     * However, this doesn't contain anything beside pts. */
    auto pktDataBuf = (Buffer*)upDecoder->GetOutput(1U);
    if (pktDataBuf && ctx.HasOutPktData()) {
      ctx.SetOutPacketData(pktDataBuf->GetDataAs<PacketData>());
    }

    auto is_seek_done = [&](DecodeContext const& ctx, int64_t pts) {
      auto seek_ctx = ctx.GetSeekContext();
      if (!seek_ctx)
        throw runtime_error("No seek context.");

      int64_t seek_pts = 0;

      if (seek_ctx->IsByNumber()) {
        seek_pts = upDemuxer->TsFromFrameNumber(seek_ctx->seek_frame);
      } else if (seek_ctx->IsByTimestamp()) {
        seek_pts = upDemuxer->TsFromTime(seek_ctx->seek_tssec);
      } else {
        throw runtime_error("Invalid seek mode.");
      }

      return (pts >= seek_pts);
    };

    /* Check if seek is done. */
    if (!use_seek) {
      loop_end = true;
    } else if (pktDataBuf) {
      auto out_pkt_data = pktDataBuf->GetDataAs<PacketData>();
      if (AV_NOPTS_VALUE == out_pkt_data->pts) {
        throw runtime_error("Decoded frame doesn't have PTS, can't seek.");
      }
      loop_end = is_seek_done(ctx, out_pkt_data->pts);
    }

    if (dmx_error) {
      cerr << "Cuvid parser exception happened." << endl;
      throw CuvidParserException();
    }

    if (dec_error && upDemuxer) {
      time_point<system_clock> then = system_clock::now();

      MuxingParams params;
      upDemuxer->GetParams(params);

      upDecoder.reset(NvdecDecodeFrame::Make(
          CudaResMgr::Instance().GetStream(gpuID),
          CudaResMgr::Instance().GetCtx(gpuID), params.videoContext.codec,
          poolFrameSize, params.videoContext.width, params.videoContext.height,
          format));

      time_point<system_clock> now = system_clock::now();
      auto duration = duration_cast<milliseconds>(now - then).count();
      cerr << "HW decoder reset time: " << duration << " milliseconds" << endl;

      throw HwResetException();
    } else if (dec_error) {
      cerr << "HW exception happened. Please reset class instance" << endl;
      throw HwResetException();
    }

    if (ctx.HasSEI()) {
      auto seiBuffer = (Buffer*)upDemuxer->GetOutput(2U);
      ctx.SetSei(seiBuffer);
    }

  } while (use_seek && !loop_end);

  if (pRawSurf) {
    ctx.SetCloneSurface(pRawSurf);
    return true;
  } else {
    return false;
  }
}

auto make_empty_surface = [](Pixel_Format pixFmt) {
  return shared_ptr<Surface>(Surface::Make(pixFmt));
};

std::map<NV_DEC_CAPS, int> PyNvDecoder::Capabilities() const {
  std::map<NV_DEC_CAPS, int> capabilities;
  capabilities.erase(capabilities.begin(), capabilities.end());

  for (int cap = BIT_DEPTH_MINUS_8; cap < NV_DEC_CAPS_NUM_ENTRIES; cap++) {
    capabilities[(NV_DEC_CAPS)cap] = upDecoder->GetCapability((NV_DEC_CAPS)cap);
  }

  return capabilities;
}

void Init_PyNvDecoder(py::module& m) {
  py::enum_<NV_DEC_CAPS>(m, "NV_DEC_CAPS")
      .value("IS_CODEC_SUPPORTED", IS_CODEC_SUPPORTED)
      .value("BIT_DEPTH_MINUS_8", BIT_DEPTH_MINUS_8)
      .value("OUTPUT_FORMAT_MASK", OUTPUT_FORMAT_MASK)
      .value("MAX_WIDTH", MAX_WIDTH)
      .value("MAX_HEIGHT", MAX_HEIGHT)
      .value("MAX_MB_COUNT", MAX_MB_COUNT)
      .value("MIN_WIDTH", MIN_WIDTH)
      .value("MIN_HEIGHT", MIN_HEIGHT)
#if CHECK_API_VERSION(11, 0)
      .value("IS_HIST_SUPPORTED", IS_HIST_SUPPORTED)
      .value("HIST_COUNT_BIT_DEPTH", HIST_COUNT_BIT_DEPTH)
      .value("HIST_COUNT_BINS", HIST_COUNT_BINS)
#endif
      .export_values();

  py::class_<PyNvDecoder, shared_ptr<PyNvDecoder>>(
      m, "PyNvDecoder", "HW-accelerated video decoder which uses Nvdec.")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec,
                    uint32_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("codec"), py::arg("gpu_id"), R"pbdoc(
        Constructor method. Initialize HW decoding session with set of particular
        parameters such as video stream resolution, pixel format and video codec.
        Use this constructor alongside external demuxer.

        :param width: video file width
        :param height: video file height
        :param format: pixel format used by codec
        :param codec: video codec to use
        :param gpu_id: what GPU to run decode on
    )pbdoc")
      .def(py::init<const string&, int, const map<string, string>&>(),
           py::arg("input"), py::arg("gpu_id"), py::arg("opts"), R"pbdoc(
        Constructor method. Initialize HW decoding section with path to input,
        GPU ID and dictionary of AVDictionary options that will be passed to built-in
        FFMpeg-based demuxer.

        :param input: path to input file
        :param gpu_id: what GPU to run decode on
        :param opts: AVDictionary options that will be passed to AVFormat context.
    )pbdoc")
      .def(py::init<const string&, int>(), py::arg("input"), py::arg("gpu_id"),
           R"pbdoc(
        Constructor method. Initialize HW decoding section with path to input,
        and GPU ID. FFMpeg-based built-in demuxer will be used.

        :param input: path to input file
        :param gpu_id: what GPU to run decode on
    )pbdoc")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, cudaVideoCodec, size_t,
                    size_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("codec"), py::arg("context"), py::arg("stream"), R"pbdoc(
        Constructor method. Initialize HW decoding session with set of particular
        parameters such as video stream resolution, pixel format, video codec,
        CUDA context and stream
        Use this constructor alongside external demuxer.

        :param width: video file width
        :param height: video file height
        :param format: pixel format used by codec
        :param codec: video codec to use
        :param context: CUDA context to use
        :param stream: CUDA stream to use
    )pbdoc")
      .def(
          py::init<const string&, size_t, size_t, const map<string, string>&>(),
          py::arg("input"), py::arg("context"), py::arg("stream"),
          py::arg("opts"), R"pbdoc(
        Constructor method. Initialize HW decoding section with path to input,
        CUDA context and stream and dictionary of AVDictionary options that will
        be passed to built-in FFMpeg-based demuxer.

        :param input: path to input file
        :param context: CUDA context to use
        :param stream: CUDA stream to use
        :param opts: AVDictionary options that will be passed to AVFormat context.
    )pbdoc")
      .def(py::init<const string&, size_t, size_t>(), py::arg("input"),
           py::arg("context"), py::arg("stream"),
           R"pbdoc(
        Constructor method. Initialize HW decoding section with path to input,
        CUDA context and stream.

        :param input: path to input file
        :param context: CUDA context to use
        :param stream: CUDA stream to use
    )pbdoc")
      .def("Width", &PyNvDecoder::Width,
           R"pbdoc(
        Return encoded video file width in pixels.
    )pbdoc")
      .def("Height", &PyNvDecoder::Height, R"pbdoc(
        Return encoded video file height in pixels.
    )pbdoc")
      .def("ColorSpace", &PyNvDecoder::GetColorSpace, R"pbdoc(
        Get color space information stored in video file.
        Please not that some video containers may not store this information.

        :return: color space information
    )pbdoc")
      .def("ColorRange", &PyNvDecoder::GetColorRange, R"pbdoc(
        Get color range information stored in video file.
        Please not that some video containers may not store this information.

        :return: color range information
    )pbdoc")
      .def("LastPacketData", &PyNvDecoder::LastPacketData, py::arg("pkt_data"),
           R"pbdoc(
        Get last packet data.

        :param pkt_data: PacketData structure.
    )pbdoc")
      .def("Framerate", &PyNvDecoder::Framerate, R"pbdoc(
        Return encoded video file framerate.
    )pbdoc")
      .def("AvgFramerate", &PyNvDecoder::AvgFramerate,
           R"pbdoc(
        Return encoded video file average framerate.
    )pbdoc")
      .def("IsVFR", &PyNvDecoder::IsVFR, R"pbdoc(
        Tell if video file has variable frame rate.

        :return: True in case video file has variable frame rate, False otherwise
    )pbdoc")
      .def("Timebase", &PyNvDecoder::Timebase,
           R"pbdoc(
        Return encoded video file time base.
    )pbdoc")
      .def("Framesize", &PyNvDecoder::Framesize,
           R"pbdoc(
        Return decoded video frame size in bytes.
    )pbdoc")
      .def("Numframes", &PyNvDecoder::Numframes,
           R"pbdoc(
        Return number of video frames in encoded video file.
        Please note that some video containers doesn't store this infomation.
    )pbdoc")
      .def("Format", &PyNvDecoder::GetPixelFormat,
           R"pbdoc(
        Return encoded video file pixel format.
    )pbdoc")
      .def("Capabilities", &PyNvDecoder::Capabilities,
           py::return_value_policy::move,
           R"pbdoc(
        Return dictionary with Nvdec capabilities.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data, nullptr,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("pkt_data"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.

        :param pkt_data: PacketData structure of decoded frame with PTS, DTS etc.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei) {
            DecodeContext ctx(&sei, nullptr, nullptr, nullptr, nullptr, false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("sei"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.

        :param sei: decoded frame SEI data
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei,
             PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, nullptr, nullptr, &out_pkt_data, nullptr,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("sei"), py::arg("pkt_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.

        :param sei: decoded frame SEI data
        :param pkt_data: PacketData structure of decoded frame with PTS, DTS etc.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei,
             SeekContext& seek_ctx) {
            DecodeContext ctx(&sei, nullptr, nullptr, nullptr, &seek_ctx,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("sei"), py::arg("seek_context"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.
        Use this function for seek + decode.

        :param sei: decoded frame SEI data
        :param seek_context: SeekContext structure with information about seek procedure
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& sei,
             SeekContext& seek_ctx, PacketData& out_pkt_data) {
            DecodeContext ctx(&sei, nullptr, nullptr, &out_pkt_data, &seek_ctx,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("sei"), py::arg("seek_context"), py::arg("pkt_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.
        Use this function for seek + decode.

        :param sei: decoded frame SEI data
        :param seek_context: SeekContext structure with information about seek procedure
        :param pkt_data: PacketData structure of decoded frame with PTS, DTS etc.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.

        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, SeekContext& seek_ctx) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, &seek_ctx,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("seek_context"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.
        Use this function for seek + decode.

        :param seek_context: SeekContext structure with information about seek procedure
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyNvDecoder> self, SeekContext& seek_ctx,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data,
                              &seek_ctx, false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("seek_context"), py::arg("pkt_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input stream.
        Video frame is returned as Surface stored in vRAM.
        Use this function for seek + decode.

        :param seek_context: SeekContext structure with information about seek procedure
        :param pkt_data: PacketData structure of decoded frame with PTS, DTS etc.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& packet) {
            DecodeContext ctx(nullptr, &packet, nullptr, nullptr, nullptr,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("packet"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from compressed video packet.
        Please note that function may not return decoded Surface.
        Use this to decode compressed packets obtained from external demuxer.

        Video frame is returned as Surface stored in vRAM.

        :param packet: encoded video packet
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, PacketData& in_packet_data,
             py::array_t<uint8_t>& packet) {
            DecodeContext ctx(nullptr, &packet, &in_packet_data, nullptr,
                              nullptr, false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("enc_packet_data"), py::arg("packet"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from compressed video packet.
        Please note that function may not return decoded Surface.
        Use this to decode compressed packets obtained from external demuxer.

        Video frame is returned as Surface stored in vRAM.

        :param enc_packet_data: PacketData structure of encoded video packet
        :param packet: encoded video packet
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, py::array_t<uint8_t>& packet,
             PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, &packet, nullptr, &out_pkt_data, nullptr,
                              false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("packet"), py::arg("pkt_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from compressed video packet.
        Please note that function may not return decoded Surface.
        Use this to decode compressed packets obtained from external demuxer.

        Video frame is returned as Surface stored in vRAM.

        :param packet: encoded video packet
        :param pkt_data: PacketData structure of decoded frame with PTS, DTS etc.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSurfaceFromPacket",
          [](shared_ptr<PyNvDecoder> self, PacketData& in_pkt_data,
             py::array_t<uint8_t>& packet, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, &packet, &in_pkt_data, &out_pkt_data,
                              nullptr, false);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("enc_packet_data"), py::arg("packet"), py::arg("pkt_data"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from compressed video packet.
        Please note that function may not return decoded Surface.
        Use this to decode compressed packets obtained from external demuxer.

        Video frame is returned as Surface stored in vRAM.

        :param enc_packet_data: PacketData structure of encoded video packet
        :param packet: encoded video packet
        :param pkt_data: PacketData structure of decoded frame with PTS, DTS etc.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "FlushSingleSurface",
          [](shared_ptr<PyNvDecoder> self) {
            DecodeContext ctx(nullptr, nullptr, nullptr, nullptr, nullptr,
                              true);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Send null input to decoder.
        Use this function in the end of decoding session to flush decoder and
        obtain those video frames which were not returned yet.

        If this method returns empty Surface it means there are no decoded frames left.

        Video frame is returned as Surface stored in vRAM.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc")
      .def(
          "FlushSingleSurface",
          [](shared_ptr<PyNvDecoder> self, PacketData& out_pkt_data) {
            DecodeContext ctx(nullptr, nullptr, nullptr, &out_pkt_data, nullptr,
                              true);
            TaskExecDetails details;
            if (self->DecodeSurface(ctx, details)) {
              return std::make_tuple(ctx.GetSurfaceMutable(), details.info);
            } else {
              return std::make_tuple(make_empty_surface(self->GetPixelFormat()),
                                     details.info);
            }
          },
          py::arg("pkt_data"), py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Send null input to decoder.
        Use this function in the end of decoding session to flush decoder and
        obtain those video frames which were not returned yet.

        If this method returns empty Surface it means there are no decoded frames left.

        Video frame is returned as Surface stored in vRAM.

        :param pkt_data: PacketData structure of decoded frame with PTS, DTS etc.
        :return: tuple, first element is the surface, second is TaskExecInfo.
    )pbdoc");
}
