// Exact row-major SDOT kernels specialized for AX1's 480x160 QKV projection.
// The 8-row arm mirrors the existing C++ topology with a fixed schedule. The
// 16-row arm trades twice the live accumulators for half as many row-block
// loops. Both return raw int32 dot products; C++ applies the identical scales.

.section __TEXT,__text,regular,pure_instructions
.p2align 5
.globl _ternlog_vocab_qkv_sdot_480x160_m4_8row
_ternlog_vocab_qkv_sdot_480x160_m4_8row:
    mov     w3, #60
L_qkv8_rows:
    movi    v0.4s, #0
    movi    v1.4s, #0
    movi    v2.4s, #0
    movi    v3.4s, #0
    movi    v4.4s, #0
    movi    v5.4s, #0
    movi    v6.4s, #0
    movi    v7.4s, #0
    mov     x4, x1
    mov     w5, #10
.p2align 5
L_qkv8_columns:
    ldr     q31, [x4], #16
    ldr     q16, [x0, #0]
    ldr     q17, [x0, #160]
    ldr     q18, [x0, #320]
    ldr     q19, [x0, #480]
    sdot    v0.4s, v16.16b, v31.16b
    sdot    v1.4s, v17.16b, v31.16b
    sdot    v2.4s, v18.16b, v31.16b
    sdot    v3.4s, v19.16b, v31.16b
    ldr     q20, [x0, #640]
    ldr     q21, [x0, #800]
    ldr     q22, [x0, #960]
    ldr     q23, [x0, #1120]
    sdot    v4.4s, v20.16b, v31.16b
    sdot    v5.4s, v21.16b, v31.16b
    sdot    v6.4s, v22.16b, v31.16b
    sdot    v7.4s, v23.16b, v31.16b
    add     x0, x0, #16
    subs    w5, w5, #1
    b.ne    L_qkv8_columns
    addv    s16, v0.4s
    addv    s17, v1.4s
    addv    s18, v2.4s
    addv    s19, v3.4s
    addv    s20, v4.4s
    addv    s21, v5.4s
    addv    s22, v6.4s
    addv    s23, v7.4s
    st1     {v16.s}[0], [x2], #4
    st1     {v17.s}[0], [x2], #4
    st1     {v18.s}[0], [x2], #4
    st1     {v19.s}[0], [x2], #4
    st1     {v20.s}[0], [x2], #4
    st1     {v21.s}[0], [x2], #4
    st1     {v22.s}[0], [x2], #4
    st1     {v23.s}[0], [x2], #4
    add     x0, x0, #1120
    subs    w3, w3, #1
    b.ne    L_qkv8_rows
    ret

.p2align 5
.globl _ternlog_vocab_qkv_sdot_480x160_m4_16row
_ternlog_vocab_qkv_sdot_480x160_m4_16row:
    mov     w3, #30
L_qkv16_rows:
    movi    v0.4s, #0
    movi    v1.4s, #0
    movi    v2.4s, #0
    movi    v3.4s, #0
    movi    v4.4s, #0
    movi    v5.4s, #0
    movi    v6.4s, #0
    movi    v7.4s, #0
    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
    mov     x4, x1
    mov     w5, #10
.p2align 5
L_qkv16_columns:
    ldr     q31, [x4], #16
    ldr     q24, [x0, #0]
    ldr     q25, [x0, #160]
    ldr     q26, [x0, #320]
    ldr     q27, [x0, #480]
    sdot    v0.4s, v24.16b, v31.16b
    sdot    v1.4s, v25.16b, v31.16b
    sdot    v2.4s, v26.16b, v31.16b
    sdot    v3.4s, v27.16b, v31.16b
    ldr     q24, [x0, #640]
    ldr     q25, [x0, #800]
    ldr     q26, [x0, #960]
    ldr     q27, [x0, #1120]
    sdot    v4.4s, v24.16b, v31.16b
    sdot    v5.4s, v25.16b, v31.16b
    sdot    v6.4s, v26.16b, v31.16b
    sdot    v7.4s, v27.16b, v31.16b
    ldr     q24, [x0, #1280]
    ldr     q25, [x0, #1440]
    ldr     q26, [x0, #1600]
    ldr     q27, [x0, #1760]
    sdot    v16.4s, v24.16b, v31.16b
    sdot    v17.4s, v25.16b, v31.16b
    sdot    v18.4s, v26.16b, v31.16b
    sdot    v19.4s, v27.16b, v31.16b
    ldr     q24, [x0, #1920]
    ldr     q25, [x0, #2080]
    ldr     q26, [x0, #2240]
    ldr     q27, [x0, #2400]
    sdot    v20.4s, v24.16b, v31.16b
    sdot    v21.4s, v25.16b, v31.16b
    sdot    v22.4s, v26.16b, v31.16b
    sdot    v23.4s, v27.16b, v31.16b
    add     x0, x0, #16
    subs    w5, w5, #1
    b.ne    L_qkv16_columns
    addv    s24, v0.4s
    str     s24, [x2], #4
    addv    s24, v1.4s
    str     s24, [x2], #4
    addv    s24, v2.4s
    str     s24, [x2], #4
    addv    s24, v3.4s
    str     s24, [x2], #4
    addv    s24, v4.4s
    str     s24, [x2], #4
    addv    s24, v5.4s
    str     s24, [x2], #4
    addv    s24, v6.4s
    str     s24, [x2], #4
    addv    s24, v7.4s
    str     s24, [x2], #4
    addv    s24, v16.4s
    str     s24, [x2], #4
    addv    s24, v17.4s
    str     s24, [x2], #4
    addv    s24, v18.4s
    str     s24, [x2], #4
    addv    s24, v19.4s
    str     s24, [x2], #4
    addv    s24, v20.4s
    str     s24, [x2], #4
    addv    s24, v21.4s
    str     s24, [x2], #4
    addv    s24, v22.4s
    str     s24, [x2], #4
    addv    s24, v23.4s
    str     s24, [x2], #4
    add     x0, x0, #2400
    subs    w3, w3, #1
    b.ne    L_qkv16_rows
    ret
