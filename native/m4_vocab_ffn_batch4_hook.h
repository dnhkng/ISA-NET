#pragma once

#include <cstdint>

struct TernlogFfnBatchRequest {
    float* output;
    std::int32_t* raw_sums;
    const std::uint8_t* final_state;
    const std::int8_t* pooled;
    const std::int8_t* weights;
    const float* decoder;
    const float* bias;
    const float* correction_scales;
};

using TernlogFfnBatchSubmit = bool (*)(const TernlogFfnBatchRequest&);

extern "C" thread_local TernlogFfnBatchSubmit ternlog_ffn_batch_submit;
