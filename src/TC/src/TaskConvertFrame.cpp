#include "Tasks.hpp"
#include "Utils.hpp"
#include <memory>
#include <stdexcept>

extern "C" {
#include <libswscale/swscale.h>
}

namespace VPF {
struct ConvertFrame_Impl {
  const AVPixelFormat m_src_fmt, m_dst_fmt;
  size_t m_width, m_height;

  std::shared_ptr<SwsContext> m_ctx = nullptr;

  ConvertFrame_Impl(uint32_t width, uint32_t height, Pixel_Format in_Format,
                    Pixel_Format out_Format)
      : m_src_fmt(toFfmpegPixelFormat(in_Format)),
        m_dst_fmt(toFfmpegPixelFormat(out_Format)), m_width(width),
        m_height(height) {
    m_ctx.reset(sws_getContext(m_width, m_height, m_src_fmt, width, height,
                               m_dst_fmt, SWS_BILINEAR, nullptr, nullptr,
                               nullptr),
                [](auto* p) { sws_freeContext(p); });

    if (!m_ctx) {
      throw std::runtime_error("ConvertFrame: sws_getContext failed");
    }
  }
};
}; // namespace VPF

ConvertFrame::~ConvertFrame() { delete pImpl; }

ConvertFrame::ConvertFrame(uint32_t width, uint32_t height,
                           Pixel_Format src_fmt, Pixel_Format dst_fmt)
    : Task("FfmpegConvertFrame", ConvertFrame::numInputs,
           ConvertFrame::numOutputs) {

  pImpl = new ConvertFrame_Impl(width, height, src_fmt, dst_fmt);
}

ConvertFrame* ConvertFrame::Make(uint32_t width, uint32_t height,
                                 Pixel_Format m_src_fmt,
                                 Pixel_Format m_dst_fmt) {
  return new ConvertFrame(width, height, m_src_fmt, m_dst_fmt);
}

TaskExecDetails ConvertFrame::Run() {
  ClearOutputs();
  try {
    auto src_buf = dynamic_cast<Buffer*>(GetInput(0));
    if (!src_buf) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                             TaskExecInfo::INVALID_INPUT, "empty src");
    }

    auto dst_buf = dynamic_cast<Buffer*>(GetInput(1));
    if (!dst_buf) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                             TaskExecInfo::INVALID_INPUT, "empty dst");
    }

    auto ctx_buf = dynamic_cast<Buffer*>(GetInput(2));
    if (!ctx_buf) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                             TaskExecInfo::INVALID_INPUT, "empty cc_ctx");
    }

    auto src_frame =
        asAVFrame(src_buf, pImpl->m_width, pImpl->m_height, pImpl->m_src_fmt);

    auto dst_frame =
        asAVFrame(dst_buf, pImpl->m_width, pImpl->m_height, pImpl->m_dst_fmt);

    auto pCtx = ctx_buf->GetDataAs<ColorspaceConversionContext>();

    auto const colorSpace = toFfmpegColorSpace(pCtx->color_space);
    auto const isJpegRange =
        (toFfmpegColorRange(pCtx->color_range) == AVCOL_RANGE_JPEG);
    auto const brightness = 0U, contrast = 1U << 16U, saturation = 1U << 16U;
    auto err = sws_setColorspaceDetails(
        pImpl->m_ctx.get(), sws_getCoefficients(colorSpace), isJpegRange,
        sws_getCoefficients(colorSpace), isJpegRange, brightness, contrast,
        saturation);
    if (err < 0) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                             TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS,
                             "unsupported cconv params");
    }

    err = sws_scale(pImpl->m_ctx.get(), src_frame->data, src_frame->linesize, 0,
                    pImpl->m_height, dst_frame->data, dst_frame->linesize);
    if (err < 0) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                             TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS,
                             AvErrorToString(err));
    }

    SetOutput(dst_buf, 0U);

    return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                           TaskExecInfo::SUCCESS);
  } catch (std::exception& e) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                           e.what());
  } catch (...) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                           "unknown exception");
  }
}