// Exact SDOT kernels for the selected 192x160 full-ticket QKV projection.
// Weights are already packed as 24 ticket-major 8-row blocks. Each block has
// ten consecutive 16-byte input tickets and 8x16 bytes of weights per ticket.
// Both functions return one raw int32 dot-product sum per output row.

.section __TEXT,__text,regular,pure_instructions

.p2align 5
.globl _ternlog_vocab_qkv_packed_sdot_192x160_m4_8row
_ternlog_vocab_qkv_packed_sdot_192x160_m4_8row:
    mov     w3, #24
L_packed_qkv8_rows:
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
L_packed_qkv8_columns:
    ldr     q31, [x4], #16
    ldp     q16, q17, [x0], #32
    ldp     q18, q19, [x0], #32
    sdot    v0.4s, v16.16b, v31.16b
    sdot    v1.4s, v17.16b, v31.16b
    sdot    v2.4s, v18.16b, v31.16b
    sdot    v3.4s, v19.16b, v31.16b
    ldp     q20, q21, [x0], #32
    ldp     q22, q23, [x0], #32
    sdot    v4.4s, v20.16b, v31.16b
    sdot    v5.4s, v21.16b, v31.16b
    sdot    v6.4s, v22.16b, v31.16b
    sdot    v7.4s, v23.16b, v31.16b
    subs    w5, w5, #1
    b.ne    L_packed_qkv8_columns
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
    subs    w3, w3, #1
    b.ne    L_packed_qkv8_rows
    ret

.p2align 5
.globl _ternlog_vocab_qkv_packed_sdot_192x160_m4_16row
_ternlog_vocab_qkv_packed_sdot_192x160_m4_16row:
    mov     w3, #12
    mov     x6, x0
    add     x7, x0, #1280
L_packed_qkv16_rows:
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
L_packed_qkv16_columns:
    ldr     q31, [x4], #16
    ldp     q24, q25, [x6], #32
    ldp     q26, q27, [x6], #32
    sdot    v0.4s, v24.16b, v31.16b
    sdot    v1.4s, v25.16b, v31.16b
    sdot    v2.4s, v26.16b, v31.16b
    sdot    v3.4s, v27.16b, v31.16b
    ldp     q24, q25, [x6], #32
    ldp     q26, q27, [x6], #32
    sdot    v4.4s, v24.16b, v31.16b
    sdot    v5.4s, v25.16b, v31.16b
    sdot    v6.4s, v26.16b, v31.16b
    sdot    v7.4s, v27.16b, v31.16b
    ldp     q24, q25, [x7], #32
    ldp     q26, q27, [x7], #32
    sdot    v16.4s, v24.16b, v31.16b
    sdot    v17.4s, v25.16b, v31.16b
    sdot    v18.4s, v26.16b, v31.16b
    sdot    v19.4s, v27.16b, v31.16b
    ldp     q24, q25, [x7], #32
    ldp     q26, q27, [x7], #32
    sdot    v20.4s, v24.16b, v31.16b
    sdot    v21.4s, v25.16b, v31.16b
    sdot    v22.4s, v26.16b, v31.16b
    sdot    v23.4s, v27.16b, v31.16b
    subs    w5, w5, #1
    b.ne    L_packed_qkv16_columns
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
    mov     x6, x7
    add     x7, x7, #1280
    subs    w3, w3, #1
    b.ne    L_packed_qkv16_rows
    ret
