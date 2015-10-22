/************************************************************************
 *
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * $Id£∫ hl_arm_blas_kernel.cc,v 1.0 2014/10/22 14:54:44  Exp $
 *
 ************************************************************************/
/**
 * @file hl_arm_blas_kernel.cc
 * @author libo
 * @date 2014/10/22 14:54:44
 * @version $Revision: 1.0$
 * @brief intrinsics for blas functions
 *
 **/
#include <math.h>
#include "cblas_hppl.h"
/**
 * @brief   adjust∫Ø ˝:src * scale + bias.
 *
 * @param   dst[out]        ‰≥ˆæÿ’Û(Elem_t, m*n).
 *          src[in]         ‰»Îæÿ’Û(int, m*n).
 *          bias_scale[in] ∆´÷√æÿ’Û∫Õ≥ﬂ∂»æÿ’Û÷∏’Î(Elem_t).
 *          dimM[in]        ‰»Îæÿ’ÛŒ¨ ˝∫Õ¡– ˝÷∏’Î(int).
 *
 * @return  void
 *
 * @note    not implemented
 */
void neon_adjust_i2f_f32(Elem_t * dst,
                         int * src,
                         Elem_t ** bias_scale,
                         int * dim)
{
    // TODO:
}

/**
 * @brief   char¿‡–Õæÿ’ÛA”Îchar¿‡–Õæÿ’ÛBœ‡≥À.
 *
 * @param   dst[out]     ‰≥ˆæÿ’ÛC.
 *          src1[in]     ‰»Îæÿ’ÛA.
 *          src2[in]     ‰»Îæÿ’ÛB.
 *          mkn[in]     æÿ’Ûµƒ∏˜∏ˆŒ¨ ˝.
 *
 * @return  void
 */
void neon_matrixmul_4x8_c8_i32(int32_t * dst,
                               int8_t * src1,
                               int8_t * src2,
                               int *mkn)
{
    int m = mkn[0];
    int k = mkn[1];
    int n = mkn[2];
    
    for (int i = 0; i < m; i += 4)
    {
        for (int j = 0; j < n; j += 8)
        {
            int32x4_t q0 = {0};
            int32x4_t q1 = {0};
            int32x4_t q2 = {0};
            int32x4_t q3 = {0};
            int32x4_t q4 = {0};
            int32x4_t q5 = {0};
            int32x4_t q6 = {0};
            int32x4_t q7 = {0};
            int32x4_t q8, q9, q10;
            for (int l = 0; l < k; l += 4)
            {
                q8 = vld1q_lane_s32((int32_t*)(src1      ), q8, 0);
                q8 = vld1q_lane_s32((int32_t*)(src1 + k  ), q8, 1);
                q8 = vld1q_lane_s32((int32_t*)(src1 + k*2), q8, 2);
                q8 = vld1q_lane_s32((int32_t*)(src1 + k*3), q8, 3);
                int16x8_t a00 = vmovl_s8(vreinterpret_s8_s32 \
                                         (vget_low_s32(q8)));
                int16x8_t a01 = vmovl_s8(vreinterpret_s8_s32 \
                                         (vget_high_s32(q8)));
                q9  = vld1q_s32((int32_t*)(src2     ));
                q10 = vld1q_s32((int32_t*)(src2 + 16));
                int16x8_t b00 = vmovl_s8(vreinterpret_s8_s32 \
                                         (vget_low_s32(q9)));
                int16x8_t b01 = vmovl_s8(vreinterpret_s8_s32 \
                                         (vget_high_s32(q9)));
                int16x8_t b02 = vmovl_s8(vreinterpret_s8_s32 \
                                         (vget_low_s32(q10)));
                int16x8_t b03 = vmovl_s8(vreinterpret_s8_s32 \
                                         (vget_high_s32(q10)));
                
                int16x4_t d8  = vget_low_s16(a00);
                int16x4_t d9  = vget_high_s16(a00);
                int16x4_t d10 = vget_low_s16(a01);
                int16x4_t d11 = vget_high_s16(a01);
                
                int16x4_t d0 = vget_low_s16(b00);
                int16x4_t d1 = vget_high_s16(b00);
                int16x4_t d2 = vget_low_s16(b01);
                int16x4_t d3 = vget_high_s16(b01);
                int16x4_t d4 = vget_low_s16(b02);
                int16x4_t d5 = vget_high_s16(b02);
                int16x4_t d6 = vget_low_s16(b03);
                int16x4_t d7 = vget_high_s16(b03);
                
                q0 = vmlal_lane_s16(q0, d0, d8,  0);
                q1 = vmlal_lane_s16(q1, d1, d8,  0);
                q2 = vmlal_lane_s16(q2, d0, d9,  0);
                q3 = vmlal_lane_s16(q3, d1, d9,  0);
                q4 = vmlal_lane_s16(q4, d0, d10, 0);
                q5 = vmlal_lane_s16(q5, d1, d10, 0);
                q6 = vmlal_lane_s16(q6, d0, d11, 0);
                q7 = vmlal_lane_s16(q7, d1, d11, 0);
                
                q0 = vmlal_lane_s16(q0, d2, d8,  1);
                q1 = vmlal_lane_s16(q1, d3, d8,  1);
                q2 = vmlal_lane_s16(q2, d2, d9,  1);
                q3 = vmlal_lane_s16(q3, d3, d9,  1);
                q4 = vmlal_lane_s16(q4, d2, d10, 1);
                q5 = vmlal_lane_s16(q5, d3, d10, 1);
                q6 = vmlal_lane_s16(q6, d2, d11, 1);
                q7 = vmlal_lane_s16(q7, d3, d11, 1);
                
                q0 = vmlal_lane_s16(q0, d4, d8,  2);
                q1 = vmlal_lane_s16(q1, d5, d8,  2);
                q2 = vmlal_lane_s16(q2, d4, d9,  2);
                q3 = vmlal_lane_s16(q3, d5, d9,  2);
                q4 = vmlal_lane_s16(q4, d4, d10, 2);
                q5 = vmlal_lane_s16(q5, d5, d10, 2);
                q6 = vmlal_lane_s16(q6, d4, d11, 2);
                q7 = vmlal_lane_s16(q7, d5, d11, 2);
                
                q0 = vmlal_lane_s16(q0, d6, d8,  3);
                q1 = vmlal_lane_s16(q1, d7, d8,  3);
                q2 = vmlal_lane_s16(q2, d6, d9,  3);
                q3 = vmlal_lane_s16(q3, d7, d9,  3);
                q4 = vmlal_lane_s16(q4, d6, d10, 3);
                q5 = vmlal_lane_s16(q5, d7, d10, 3);
                q6 = vmlal_lane_s16(q6, d6, d11, 3);
                q7 = vmlal_lane_s16(q7, d7, d11, 3);
                
                src1 += 4;
                src2 += 32;
            }// end for l
            vst1q_s32((dst      ), q0); vst1q_s32((dst + 4      ), q1);
            vst1q_s32((dst + n  ), q2); vst1q_s32((dst + 4 + n  ), q3);
            vst1q_s32((dst + n*2), q4); vst1q_s32((dst + 4 + n*2), q5);
            vst1q_s32((dst + n*3), q6); vst1q_s32((dst + 4 + n*3), q7);
            dst  += 8;
            src1 -= k;
        }// end for j
        dst  += n*3;
        src1 += k*4;
        src2 -= k*n;
    }// end for i
}

/**
 * @brief   uchar¿‡–Õæÿ’ÛA”Îchar¿‡–Õæÿ’ÛBœ‡≥À.
 *
 * @param   dst[out]     ‰≥ˆæÿ’ÛC.
 *          src1[in]     ‰»Îæÿ’ÛA.
 *          src2[in]     ‰»Îæÿ’ÛB.
 *          mkn[in]     æÿ’Ûµƒ∏˜∏ˆŒ¨ ˝.
 *
 * @return  void
 */
void neon_matrixmul_4x8_uc8_i32(int32_t * dst,
                                uint8_t * src1,
                                int8_t * src2,
                                int *mkn)
{
    int m = mkn[0];
    int k = mkn[1];
    int n = mkn[2];
    
    for (int i = 0; i < m; i += 4)
    {
        for (int j = 0; j < n; j += 8)
        {
            int32x4_t q0 = {0};
            int32x4_t q1 = {0};
            int32x4_t q2 = {0};
            int32x4_t q3 = {0};
            int32x4_t q4 = {0};
            int32x4_t q5 = {0};
            int32x4_t q6 = {0};
            int32x4_t q7 = {0};
            int32x4_t q8, q9, q10;
            for (int l = 0; l < k; l += 4)
            {
                q8 = vld1q_lane_s32((int32_t*)(src1      ), q8, 0);
                q8 = vld1q_lane_s32((int32_t*)(src1 + k  ), q8, 1);
                q8 = vld1q_lane_s32((int32_t*)(src1 + k*2), q8, 2);
                q8 = vld1q_lane_s32((int32_t*)(src1 + k*3), q8, 3);
                int16x8_t a00 = vreinterpretq_s16_u16(vmovl_u8 \
                                                      (vreinterpret_u8_s32(vget_low_s32(q8))));
                int16x8_t a01 = vreinterpretq_s16_u16(vmovl_u8 \
                                                      (vreinterpret_u8_s32(vget_high_s32(q8))));
                q9  = vld1q_s32((int32_t*)(src2     ));
                q10 = vld1q_s32((int32_t*)(src2 + 16));
                int16x8_t b00 = vmovl_s8 (vreinterpret_s8_s32 \
                                          (vget_low_s32(q9)));
                int16x8_t b01 = vmovl_s8 (vreinterpret_s8_s32 \
                                          (vget_high_s32(q9)));
                int16x8_t b02 = vmovl_s8 (vreinterpret_s8_s32 \
                                          (vget_low_s32(q10)));
                int16x8_t b03 = vmovl_s8 (vreinterpret_s8_s32 \
                                          (vget_high_s32(q10)));
                
                int16x4_t d8  = vget_low_s16(a00);
                int16x4_t d9  = vget_high_s16(a00);
                int16x4_t d10 = vget_low_s16(a01);
                int16x4_t d11 = vget_high_s16(a01);
                
                int16x4_t d0 = vget_low_s16(b00);
                int16x4_t d1 = vget_high_s16(b00);
                int16x4_t d2 = vget_low_s16(b01);
                int16x4_t d3 = vget_high_s16(b01);
                int16x4_t d4 = vget_low_s16(b02);
                int16x4_t d5 = vget_high_s16(b02);
                int16x4_t d6 = vget_low_s16(b03);
                int16x4_t d7 = vget_high_s16(b03);
                
                q0 = vmlal_lane_s16(q0, d0, d8,  0);
                q1 = vmlal_lane_s16(q1, d1, d8,  0);
                q2 = vmlal_lane_s16(q2, d0, d9,  0);
                q3 = vmlal_lane_s16(q3, d1, d9,  0);
                q4 = vmlal_lane_s16(q4, d0, d10, 0);
                q5 = vmlal_lane_s16(q5, d1, d10, 0);
                q6 = vmlal_lane_s16(q6, d0, d11, 0);
                q7 = vmlal_lane_s16(q7, d1, d11, 0);
                
                q0 = vmlal_lane_s16(q0, d2, d8,  1);
                q1 = vmlal_lane_s16(q1, d3, d8,  1);
                q2 = vmlal_lane_s16(q2, d2, d9,  1);
                q3 = vmlal_lane_s16(q3, d3, d9,  1);
                q4 = vmlal_lane_s16(q4, d2, d10, 1);
                q5 = vmlal_lane_s16(q5, d3, d10, 1);
                q6 = vmlal_lane_s16(q6, d2, d11, 1);
                q7 = vmlal_lane_s16(q7, d3, d11, 1);
                
                q0 = vmlal_lane_s16(q0, d4, d8,  2);
                q1 = vmlal_lane_s16(q1, d5, d8,  2);
                q2 = vmlal_lane_s16(q2, d4, d9,  2);
                q3 = vmlal_lane_s16(q3, d5, d9,  2);
                q4 = vmlal_lane_s16(q4, d4, d10, 2);
                q5 = vmlal_lane_s16(q5, d5, d10, 2);
                q6 = vmlal_lane_s16(q6, d4, d11, 2);
                q7 = vmlal_lane_s16(q7, d5, d11, 2);
                
                q0 = vmlal_lane_s16(q0, d6, d8,  3);
                q1 = vmlal_lane_s16(q1, d7, d8,  3);
                q2 = vmlal_lane_s16(q2, d6, d9,  3);
                q3 = vmlal_lane_s16(q3, d7, d9,  3);
                q4 = vmlal_lane_s16(q4, d6, d10, 3);
                q5 = vmlal_lane_s16(q5, d7, d10, 3);
                q6 = vmlal_lane_s16(q6, d6, d11, 3);
                q7 = vmlal_lane_s16(q7, d7, d11, 3);
                
                src1 += 4;
                src2 += 32;
            }// end for l
            vst1q_s32((dst      ), q0); vst1q_s32((dst + 4      ), q1);
            vst1q_s32((dst + n  ), q2); vst1q_s32((dst + 4 + n  ), q3);
            vst1q_s32((dst + n*2), q4); vst1q_s32((dst + 4 + n*2), q5);
            vst1q_s32((dst + n*3), q6); vst1q_s32((dst + 4 + n*3), q7);
            dst  += 8;
            src1 -= k;
        }// end for j
        dst  += n*3;
        src1 += k*4;
        src2 -= k*n;
    }// end for i
}

/**
 * @brief   short¿‡–Õæÿ’ÛA”Îchar¿‡–Õæÿ’ÛBœ‡≥À.
 *
 * @param   dst[out]     ‰≥ˆæÿ’ÛC.
 *          src1[in]     ‰»Îæÿ’ÛA.
 *          src2[in]     ‰»Îæÿ’ÛB.
 *          mkn[in]     æÿ’Ûµƒ∏˜∏ˆŒ¨ ˝.
 *
 * @return  void
 *
 * @note    not implemented
 */
void neon_matrixmul_4x8_s16_i32(int32_t * dst,
                                int16_t * src1,
                                int8_t * src2,
                                int *mkn)
{
    // TODO:
}

/**
 * @brief   vector_mul_part_matrix(fixed)
 *
 * @param   dst[out]     the output vector(1*n)
 *          src1[in]     the input  vector(1*k)
 *          src2[in]     the input  matrix(k*N)
 *          cand[in]     the index  of  matrix
 *          kn[in]       DIM_K & DIM_N
 *
 * @return  void
 */
void neon_vectormulmatrix_pchar(int32_t *dst,
                                int8_t *src1,
                                int8_t *src2,
                                int    *cand,
                                int *kn)
{
    int j ,l, i;
    int k = kn[0];
    int n = kn[1];
    for (j = 0; j <= n - 4; j += 4) {
        int loc0 = cand[j];
        int loc1 = cand[j+1];
        int loc2 = cand[j+2];
        int loc3 = cand[j+3];
        int8x8_t a00 = vld1_s8(src1);
        int8x8_t b00 = vld1_s8(src2 + (loc0*k));
        int8x8_t b01 = vld1_s8(src2 + (loc1*k));
        int8x8_t b02 = vld1_s8(src2 + (loc2*k));
        int8x8_t b03 = vld1_s8(src2 + (loc3*k));
        
        int16x8_t c00  = vmull_s8(a00, b00);
        int16x8_t c10  = vmull_s8(a00, b01);
        int16x8_t c20  = vmull_s8(a00, b02);
        int16x8_t c30  = vmull_s8(a00, b03);
        
        int32x4_t f00  = vpaddlq_s16(c00);
        int32x4_t f10  = vpaddlq_s16(c10);
        int32x4_t f20  = vpaddlq_s16(c20);
        int32x4_t f30  = vpaddlq_s16(c30);
        
        src1 += 8;
        src2 += 8;
        for (l = 8; l <= k - 8; l += 8)
        {
            a00 = vld1_s8(src1);
            b00 = vld1_s8(src2 + (loc0*k));
            b01 = vld1_s8(src2 + (loc1*k));
            b02 = vld1_s8(src2 + (loc2*k));
            b03 = vld1_s8(src2 + (loc3*k));
            
            c00  = vmull_s8(a00, b00);
            f00  = vaddq_s32(f00, vpaddlq_s16(c00));
            c10  = vmull_s8(a00, b01);
            f10  = vaddq_s32(f10, vpaddlq_s16(c10));
            c20  = vmull_s8(a00, b02);
            f20  = vaddq_s32(f20, vpaddlq_s16(c20));
            c30  = vmull_s8(a00, b03);
            f30  = vaddq_s32(f30, vpaddlq_s16(c30));
            
            src1 += 8;
            src2 += 8;
        }
        
        int32x2_t e00 = vpadd_s32(vget_low_s32(f00), vget_high_s32(f00));
        int32x2_t e01 = vpadd_s32(vget_low_s32(f10), vget_high_s32(f10));
        int32x2_t e02 = vpadd_s32(vget_low_s32(f20), vget_high_s32(f20));
        int32x2_t e03 = vpadd_s32(vget_low_s32(f30), vget_high_s32(f30));
        
        e00 = vpadd_s32(e00, e01);
        e02 = vpadd_s32(e02, e03);
        
        vst1q_s32(dst, vcombine_s32(e00, e02));
        dst  += 4;
        src1 -= k;
        src2 -= k;
    }
    for (; j < n; j++) {
        int loc0 = cand[j];
        int8x8_t a00 = vld1_s8(src1);
        int8x8_t b00 = vld1_s8(src2 + (loc0*k));
        
        int16x8_t c00  = vmull_s8(a00, b00);
        int32x4_t f00  = vpaddlq_s16(c00);
        
        src1 += 8;
        src2 += 8;
        for (l = 8; l <= k - 8; l += 8)
        {
            a00 = vld1_s8(src1);
            b00 = vld1_s8(src2 + (loc0*k));
            
            c00  = vmull_s8(a00, b00);
            f00  = vaddq_s32(f00, vpaddlq_s16(c00));
            
            src1 += 8;
            src2 += 8;
        }
        int32x2_t e00 = vpadd_s32(vget_low_s32(f00), vget_high_s32(f00));
        
        e00 = vpadd_s32(e00, e00);
        *dst++  = *(int32_t *)(&e00);
        src1 -= k;
        src2 -= k;
    }
}

/**
 * @brief   vector_mul_matrix(fixed)
 *
 * @param   dst[out]     the output vector(1*n)
 *          src1[in]     the input  vector(1*k)
 *          src2[in]     the input  matrix(k*n)
 *          kn[in]       DIM_K & DIM_N
 *
 * @return  void
 */
void neon_vectormulmatrix_char(int32_t *dst,
                               int8_t *src1,
                               int8_t *src2,
                               int *kn)
{
    int l, i;
    int k = kn[0];
    int n = kn[1];
    int j = 0;
    for (j = 0; j <= n - 4; j += 4) {
        int8x8_t a00 = vld1_s8(src1);
        int8x8_t b00 = vld1_s8(src2 + (j*k));
        int8x8_t b01 = vld1_s8(src2 + ((j+1)*k));
        int8x8_t b02 = vld1_s8(src2 + ((j+2)*k));
        int8x8_t b03 = vld1_s8(src2 + ((j+3)*k));
        
        int16x8_t c00  = vmull_s8(a00, b00);
        int16x8_t c10  = vmull_s8(a00, b01);
        int16x8_t c20  = vmull_s8(a00, b02);
        int16x8_t c30  = vmull_s8(a00, b03);
        
        int32x4_t f00  = vpaddlq_s16(c00);
        int32x4_t f10  = vpaddlq_s16(c10);
        int32x4_t f20  = vpaddlq_s16(c20);
        int32x4_t f30  = vpaddlq_s16(c30);
        
        src1 += 8;
        src2 += 8;
        for (l = 8; l <= k - 8; l += 8)
        {
            a00 = vld1_s8(src1);
            b00 = vld1_s8(src2 + (j*k));
            b01 = vld1_s8(src2 + ((j+1)*k));
            b02 = vld1_s8(src2 + ((j+2)*k));
            b03 = vld1_s8(src2 + ((j+3)*k));
            
            c00  = vmull_s8(a00, b00);
            f00  = vaddq_s32(f00, vpaddlq_s16(c00));
            c10  = vmull_s8(a00, b01);
            f10  = vaddq_s32(f10, vpaddlq_s16(c10));
            c20  = vmull_s8(a00, b02);
            f20  = vaddq_s32(f20, vpaddlq_s16(c20));
            c30  = vmull_s8(a00, b03);
            f30  = vaddq_s32(f30, vpaddlq_s16(c30));
            
            src1 += 8;
            src2 += 8;
        }
        
        int32x2_t e00 = vpadd_s32(vget_low_s32(f00), vget_high_s32(f00));
        int32x2_t e01 = vpadd_s32(vget_low_s32(f10), vget_high_s32(f10));
        int32x2_t e02 = vpadd_s32(vget_low_s32(f20), vget_high_s32(f20));
        int32x2_t e03 = vpadd_s32(vget_low_s32(f30), vget_high_s32(f30));
        
        e00 = vpadd_s32(e00, e01);
        e02 = vpadd_s32(e02, e03);
        
        vst1q_s32(dst, vcombine_s32(e00, e02));
        dst  += 4;
        src1 -= k;
        src2 -= k;
    }
    for (; j < n; j++) {
        int8x8_t a00 = vld1_s8(src1);
        int8x8_t b00 = vld1_s8(src2 + (j*k));
        
        int16x8_t c00  = vmull_s8(a00, b00);
        int32x4_t f00  = vpaddlq_s16(c00);
        
        src1 += 8;
        src2 += 8;
        for (l = 8; l <= k - 8; l += 8)
        {
            a00 = vld1_s8(src1);
            b00 = vld1_s8(src2 + (j*k));
            c00  = vmull_s8(a00, b00);
            f00  = vaddq_s32(f00, vpaddlq_s16(c00));
            src1 += 8;
            src2 += 8;
            
        }
        int32x2_t e00 = vpadd_s32(vget_low_s32(f00), vget_high_s32(f00));
        
        e00 = vpadd_s32(e00, e00);
        *dst++  = *(int32_t *)(&e00);
        src1 -= k;
        src2 -= k;
    }
}

