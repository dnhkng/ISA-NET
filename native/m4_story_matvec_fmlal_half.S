// Row-major eight-row M4 matvec for pre-rounded FP16 activations.
// Arithmetic and accumulation order match m4_story_matvec_fmlal.S exactly;
// activation conversion is hoisted out of the row-block loop.

.section __TEXT,__text,regular,pure_instructions
.p2align 5
.globl _ternlog_matvec_f16input_fmlal_m4
_ternlog_matvec_f16input_fmlal_m4:
L_half_row_block:
    lsl     x6, x5, #1
    mov     x8, x0
    add     x9, x8, x6
    add     x10, x9, x6
    add     x11, x10, x6
    add     x12, x11, x6
    add     x13, x12, x6
    add     x14, x13, x6
    add     x15, x14, x6
    mov     x16, x2
    mov     w17, w5

    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
    movi    v24.4s, #0
    movi    v25.4s, #0
    movi    v26.4s, #0
    movi    v27.4s, #0
    movi    v28.4s, #0
    movi    v29.4s, #0
    movi    v30.4s, #0
    movi    v31.4s, #0

.p2align 5
L_half_column_loop:
    ldr     q2, [x16], #16

    ldr     q3, [x8], #16
    ldr     q4, [x9], #16
    fmlal   v16.4s, v2.4h, v3.4h
    fmlal2  v17.4s, v2.4h, v3.4h
    fmlal   v18.4s, v2.4h, v4.4h
    fmlal2  v19.4s, v2.4h, v4.4h

    ldr     q3, [x10], #16
    ldr     q4, [x11], #16
    fmlal   v20.4s, v2.4h, v3.4h
    fmlal2  v21.4s, v2.4h, v3.4h
    fmlal   v22.4s, v2.4h, v4.4h
    fmlal2  v23.4s, v2.4h, v4.4h

    ldr     q3, [x12], #16
    ldr     q4, [x13], #16
    fmlal   v24.4s, v2.4h, v3.4h
    fmlal2  v25.4s, v2.4h, v3.4h
    fmlal   v26.4s, v2.4h, v4.4h
    fmlal2  v27.4s, v2.4h, v4.4h

    ldr     q3, [x14], #16
    ldr     q4, [x15], #16
    fmlal   v28.4s, v2.4h, v3.4h
    fmlal2  v29.4s, v2.4h, v3.4h
    fmlal   v30.4s, v2.4h, v4.4h
    fmlal2  v31.4s, v2.4h, v4.4h

    subs    w17, w17, #8
    b.ne    L_half_column_loop

    fadd    v16.4s, v16.4s, v17.4s
    fadd    v18.4s, v18.4s, v19.4s
    fadd    v20.4s, v20.4s, v21.4s
    fadd    v22.4s, v22.4s, v23.4s
    fadd    v24.4s, v24.4s, v25.4s
    fadd    v26.4s, v26.4s, v27.4s
    fadd    v28.4s, v28.4s, v29.4s
    fadd    v30.4s, v30.4s, v31.4s
    faddp   v16.4s, v16.4s, v16.4s
    faddp   v18.4s, v18.4s, v18.4s
    faddp   v20.4s, v20.4s, v20.4s
    faddp   v22.4s, v22.4s, v22.4s
    faddp   v24.4s, v24.4s, v24.4s
    faddp   v26.4s, v26.4s, v26.4s
    faddp   v28.4s, v28.4s, v28.4s
    faddp   v30.4s, v30.4s, v30.4s
    faddp   s16, v16.2s
    faddp   s18, v18.2s
    faddp   s20, v20.2s
    faddp   s22, v22.2s
    faddp   s24, v24.2s
    faddp   s26, v26.2s
    faddp   s28, v28.2s
    faddp   s30, v30.2s

    ins     v0.s[0], v16.s[0]
    ins     v0.s[1], v18.s[0]
    ins     v0.s[2], v20.s[0]
    ins     v0.s[3], v22.s[0]
    ins     v1.s[0], v24.s[0]
    ins     v1.s[1], v26.s[0]
    ins     v1.s[2], v28.s[0]
    ins     v1.s[3], v30.s[0]

    cbz     x1, L_half_store
    ldr     q2, [x1], #16
    fcvtl   v3.4s, v2.4h
    fcvtl2  v4.4s, v2.8h
    fadd    v0.4s, v0.4s, v3.4s
    fadd    v1.4s, v1.4s, v4.4s
L_half_store:
    stp     q0, q1, [x3], #32
    mov     x0, x15
    subs    w4, w4, #8
    b.ne    L_half_row_block
    ret
