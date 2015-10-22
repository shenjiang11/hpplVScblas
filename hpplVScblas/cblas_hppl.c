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
 * @brief   vector scale: A[] = alpha * B[].
 *
 * @param   dst[out]    the result matrix A.
 *          src[in]     the input matrix B.
 *          alpha[in]   scale of B.
 *          elemCnt[in] number of elements to calc.
 *
 * @return  void.
 */
void neon_scale(float *dst,
                const float *src,
                const float alpha,
                const int elemCnt)
{
    int i;
    for (i = 0; i <= elemCnt - 16; i += 16)
    {
        float32x4_t q0 = vld1q_f32(src + i);
        float32x4_t q1 = vld1q_f32(src + i + 4);
        float32x4_t q2 = vld1q_f32(src + i + 8);
        float32x4_t q3 = vld1q_f32(src + i + 12);
        q0 = vmulq_n_f32(q0, alpha);
        q1 = vmulq_n_f32(q1, alpha);
        q2 = vmulq_n_f32(q2, alpha);
        q3 = vmulq_n_f32(q3, alpha);
        vst1q_f32(dst + i,      q0);
        vst1q_f32(dst + i + 4,  q1);
        vst1q_f32(dst + i + 8,  q2);
        vst1q_f32(dst + i + 12, q3);
    }
    for (; i < elemCnt; i++)
    {
        dst[i] = src[i] * alpha;
    }
}

/**
 * @brief   vector scale & accu: A[] = alpha * B[] + beta * A[].
 *
 * @param   dst[out]    the accumulating matrix A.
 *          src[in]     the input matrix B.
 *          alpha[in]   scale of B.
 *          beta[in]    scale of A.
 *          elemCnt[in] number of elements to calc.
 *
 * @return  void.
 */
void neon_axpby(float *dst,
                const float *src,
                const float alpha,
                const float beta,
                const int elemCnt)
{
    int i;
    for (i = 0; i <= elemCnt - 16; i += 16)
    {
        float32x4_t q0 = vld1q_f32(src + i);
        float32x4_t q1 = vld1q_f32(src + i + 4);
        float32x4_t q2 = vld1q_f32(src + i + 8);
        float32x4_t q3 = vld1q_f32(src + i + 12);
        float32x4_t q4 = vld1q_f32(dst + i);
        float32x4_t q5 = vld1q_f32(dst + i + 4);
        float32x4_t q6 = vld1q_f32(dst + i + 8);
        float32x4_t q7 = vld1q_f32(dst + i + 12);
        q0 = vmulq_n_f32(q0, alpha);
        q1 = vmulq_n_f32(q1, alpha);
        q2 = vmulq_n_f32(q2, alpha);
        q3 = vmulq_n_f32(q3, alpha);
        q0 = vmlaq_n_f32(q0, q4, beta);
        q1 = vmlaq_n_f32(q1, q5, beta);
        q2 = vmlaq_n_f32(q2, q6, beta);
        q3 = vmlaq_n_f32(q3, q7, beta);
        vst1q_f32(dst + i,      q0);
        vst1q_f32(dst + i + 4,  q1);
        vst1q_f32(dst + i + 8,  q2);
        vst1q_f32(dst + i + 12, q3);
    }
    for (; i < elemCnt; i++)
    {
        float a = src[i] * alpha + dst[i] * beta;
        dst[i] = a;
    }
}

/**
 * @brief   Elem_t¿‡–Õæÿ’ÛA”ÎElem_t¿‡–Õæÿ’ÛBœ‡≥À.
 *
 * @param   dst[out]     ‰≥ˆæÿ’ÛC.
 *          src1[in]     ‰»Îæÿ’ÛA.
 *          src2[in]     ‰»Îæÿ’ÛB.
 *          mkn[in]     æÿ’Ûµƒ∏˜∏ˆŒ¨ ˝.
 *
 * @return  void
 */
void neon_matrixmul_4x4float(Elem_t * dst,
                             Elem_t * src1,
                             Elem_t * src2,
                             int *mkn)
{
    int m = mkn[0];
    int k = mkn[1];
    int n = mkn[2];
    
    for (int i = 0; i < m; i += 4)
    {
        for (int j = 0; j < n; j += 4)
        {
            float32x2_t d16 = {0};
            float32x2_t d17 = {0};
            float32x2_t d18 = {0};
            float32x2_t d19 = {0};
            float32x2_t d20 = {0};
            float32x2_t d21 = {0};
            float32x2_t d22 = {0};
            float32x2_t d23 = {0};
            float32x2_t d24 = {0};
            float32x2_t d25 = {0};
            float32x2_t d26 = {0};
            float32x2_t d27 = {0};
            float32x2_t d28 = {0};
            float32x2_t d29 = {0};
            float32x2_t d30 = {0};
            float32x2_t d31 = {0};
            
            for (int l = 0; l < k; l += 4)
            {
                // Matrix A
                float32x4_t q8  = vld1q_f32(src1      );
                float32x4_t q9  = vld1q_f32(src1 + k  );
                float32x4_t q10 = vld1q_f32(src1 + k*2);
                float32x4_t q11 = vld1q_f32(src1 + k*3);
                float32x2_t d0 = vget_low_f32(q8);
                float32x2_t d1 = vget_high_f32(q8);
                float32x2_t d2 = vget_low_f32(q9);
                float32x2_t d3 = vget_high_f32(q9);
                float32x2_t d4 = vget_low_f32(q10);
                float32x2_t d5 = vget_high_f32(q10);
                float32x2_t d6 = vget_low_f32(q11);
                float32x2_t d7 = vget_high_f32(q11);
                
                // Matrix B
                float32x4_t q12 = vld1q_f32(src2      );
                float32x4_t q13 = vld1q_f32(src2 + k  );
                float32x4_t q14 = vld1q_f32(src2 + k*2);
                float32x4_t q15 = vld1q_f32(src2 + k*3);
                float32x2_t d8  = vget_low_f32(q12);
                float32x2_t d9  = vget_high_f32(q12);
                float32x2_t d10 = vget_low_f32(q13);
                float32x2_t d11 = vget_high_f32(q13);
                float32x2_t d12 = vget_low_f32(q14);
                float32x2_t d13 = vget_high_f32(q14);
                float32x2_t d14 = vget_low_f32(q15);
                float32x2_t d15 = vget_high_f32(q15);
                
                d16 = vmla_f32(d16, d0, d8);
                d17 = vmla_f32(d17, d0, d10);
                d18 = vmla_f32(d18, d0, d12);
                d19 = vmla_f32(d19, d0, d14);
                d16 = vmla_f32(d16, d1, d9);
                d17 = vmla_f32(d17, d1, d11);
                d18 = vmla_f32(d18, d1, d13);
                d19 = vmla_f32(d19, d1, d15);
                
                d20 = vmla_f32(d20, d2, d8);
                d21 = vmla_f32(d21, d2, d10);
                d22 = vmla_f32(d22, d2, d12);
                d23 = vmla_f32(d23, d2, d14);
                d20 = vmla_f32(d20, d3, d9);
                d21 = vmla_f32(d21, d3, d11);
                d22 = vmla_f32(d22, d3, d13);
                d23 = vmla_f32(d23, d3, d15);
                
                d24 = vmla_f32(d24, d4, d8);
                d25 = vmla_f32(d25, d4, d10);
                d26 = vmla_f32(d26, d4, d12);
                d27 = vmla_f32(d27, d4, d14);
                d24 = vmla_f32(d24, d5, d9);
                d25 = vmla_f32(d25, d5, d11);
                d26 = vmla_f32(d26, d5, d13);
                d27 = vmla_f32(d27, d5, d15);
                
                d28 = vmla_f32(d28, d6, d8);
                d29 = vmla_f32(d29, d6, d10);
                d30 = vmla_f32(d30, d6, d12);
                d31 = vmla_f32(d31, d6, d14);
                d28 = vmla_f32(d28, d7, d9);
                d29 = vmla_f32(d29, d7, d11);
                d30 = vmla_f32(d30, d7, d13);
                d31 = vmla_f32(d31, d7, d15);
                
                src1 += 4;
                src2 += 4;
            }// end for l
            d16 = vpadd_f32(d16, d17);
            d18 = vpadd_f32(d18, d19);
            d20 = vpadd_f32(d20, d21);
            d22 = vpadd_f32(d22, d23);
            d24 = vpadd_f32(d24, d25);
            d26 = vpadd_f32(d26, d27);
            d28 = vpadd_f32(d28, d29);
            d30 = vpadd_f32(d30, d31);
            vst1q_f32(dst      , vcombine_f32(d16, d18));
            vst1q_f32(dst + n  , vcombine_f32(d20, d22));
            vst1q_f32(dst + n*2, vcombine_f32(d24, d26));
            vst1q_f32(dst + n*3, vcombine_f32(d28, d30));
            
            src1 -= k;
            src2 += k*3;
            dst  += 4;
        }// end for j
        src1 += k*4;
        src2 -= k*n;
        dst  += n*3;
    }// end for i
}

/**
 * @brief   matrix dot mul: C[] = alpha * A[] * B[] + beta * C[].
 *
 * @param   dst[out]    the result matrix C.
 *          src1[in]    the input matrix A.
 *          src2[in]    the input matrix B.
 *          alpha[in]   scale of A * B.
 *          beta[in]    scale of C.
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *
 * @return  void.
 */
void neon_dotMul(float *dst,
                 const float *src1,
                 const float *src2,
                 const float alpha,
                 const float beta,
                 const int dimM,
                 const int dimN,
                 const int leadingN)
{
    float *mat0 = (float *)src1;
    float *mat1 = (float *)src1 + leadingN;
    float *mat2 = (float *)src1 + leadingN*2;
    float *mat3 = (float *)src1 + leadingN*3;
    float *mat4 = (float *)src2;
    float *mat5 = (float *)src2 + leadingN;
    float *mat6 = (float *)src2 + leadingN*2;
    float *mat7 = (float *)src2 + leadingN*3;
    float *mat8 = dst;
    float *mat9 = dst + leadingN;
    float *matA = dst + leadingN*2;
    float *matB = dst + leadingN*3;
    int i = 0;
    for (i = 0; i <= dimM - 4; i += 4)
    {
        int j = 0;
        for (j = 0; j <= dimN - 4; j += 4)
        {
            float32x4_t q0 = vld1q_f32(mat0 + j);
            float32x4_t q1 = vld1q_f32(mat1 + j);
            float32x4_t q2 = vld1q_f32(mat2 + j);
            float32x4_t q3 = vld1q_f32(mat3 + j);
            float32x4_t q4 = vld1q_f32(mat4 + j);
            float32x4_t q5 = vld1q_f32(mat5 + j);
            float32x4_t q6 = vld1q_f32(mat6 + j);
            float32x4_t q7 = vld1q_f32(mat7 + j);
            float32x4_t q8 = vld1q_f32(mat8 + j);
            float32x4_t q9 = vld1q_f32(mat9 + j);
            float32x4_t qA = vld1q_f32(matA + j);
            float32x4_t qB = vld1q_f32(matB + j);
            q0 = vmulq_n_f32(q0, alpha);
            q1 = vmulq_n_f32(q1, alpha);
            q2 = vmulq_n_f32(q2, alpha);
            q3 = vmulq_n_f32(q3, alpha);
            q0 = vmulq_f32(q0, q4);
            q1 = vmulq_f32(q1, q5);
            q2 = vmulq_f32(q2, q6);
            q3 = vmulq_f32(q3, q7);
            q0 = vmlaq_n_f32(q0, q8, beta);
            q1 = vmlaq_n_f32(q1, q9, beta);
            q2 = vmlaq_n_f32(q2, qA, beta);
            q3 = vmlaq_n_f32(q3, qB, beta);
            vst1q_f32(mat8 + j, q0);
            vst1q_f32(mat9 + j, q1);
            vst1q_f32(matA + j, q2);
            vst1q_f32(matB + j, q3);
        }
        for (; j < dimN; j++)
        {
            float a0 = mat8[j] * beta;
            float a1 = mat9[j] * beta;
            float a2 = matA[j] * beta;
            float a3 = matB[j] * beta;
            a0 += mat0[j] * mat4[j] * alpha;
            a1 += mat1[j] * mat5[j] * alpha;
            a2 += mat2[j] * mat6[j] * alpha;
            a3 += mat3[j] * mat7[j] * alpha;
            mat8[j] = a0;
            mat9[j] = a1;
            matA[j] = a2;
            matB[j] = a3;
        }
        mat0 += leadingN * 4;
        mat1 += leadingN * 4;
        mat2 += leadingN * 4;
        mat3 += leadingN * 4;
        mat4 += leadingN * 4;
        mat5 += leadingN * 4;
        mat6 += leadingN * 4;
        mat7 += leadingN * 4;
        mat8 += leadingN * 4;
        mat9 += leadingN * 4;
        matA += leadingN * 4;
        matB += leadingN * 4;
    }
    for (; i < dimM; i++)
    {
        int j = 0;
        for (j = 0; j <= dimN - 4; j += 4)
        {
            float32x4_t q0 = vld1q_f32(mat0 + j);
            float32x4_t q4 = vld1q_f32(mat4 + j);
            float32x4_t q8 = vld1q_f32(mat8 + j);
            q0 = vmulq_n_f32(q0, alpha);
            q0 = vmulq_f32(q0, q4);
            q0 = vmlaq_n_f32(q0, q8, beta);
            vst1q_f32(mat8 + j, q0);
        }
        for (; j < dimN; j++)
        {
            float a0 = mat0[j] * mat4[j] * alpha + mat8[j] * beta;
            mat8[j] = a0;
        }
        mat0 += leadingN;
        mat4 += leadingN;
        mat8 += leadingN;
    }
}

/**
 * @brief   matrix dot mul vector: C[] = alpha * A[] * B[] + beta * C[].
 *
 * @param   dst[out]    the result matrix C.
 *          src1[in]    the input matrix A.
 *          src2[in]    the input vector B.
 *          alpha[in]   scale of A * B.
 *          beta[in]    scale of C.
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *
 * @return  void.
 */
void neon_dotMulVec(float *dst,
                    const float *src1,
                    const float *src2,
                    const float alpha,
                    const float beta,
                    const int dimM,
                    const int dimN,
                    const int leadingN)
{
    float *mat0 = (float *)src1;
    float *mat1 = (float *)src1 + leadingN;
    float *mat2 = (float *)src1 + leadingN*2;
    float *mat3 = (float *)src1 + leadingN*3;
    float *mat4 = (float *)src2;
    float *mat8 = dst;
    float *mat9 = dst + leadingN;
    float *matA = dst + leadingN*2;
    float *matB = dst + leadingN*3;
    int i = 0;
    for (i = 0; i <= dimM - 4; i += 4)
    {
        int j = 0;
        for (j = 0; j <= dimN - 4; j += 4)
        {
            float32x4_t q0 = vld1q_f32(mat0 + j);
            float32x4_t q1 = vld1q_f32(mat1 + j);
            float32x4_t q2 = vld1q_f32(mat2 + j);
            float32x4_t q3 = vld1q_f32(mat3 + j);
            float32x4_t q4 = vld1q_f32(mat4 + j);
            float32x4_t q8 = vld1q_f32(mat8 + j);
            float32x4_t q9 = vld1q_f32(mat9 + j);
            float32x4_t qA = vld1q_f32(matA + j);
            float32x4_t qB = vld1q_f32(matB + j);
            q0 = vmulq_n_f32(q0, alpha);
            q1 = vmulq_n_f32(q1, alpha);
            q2 = vmulq_n_f32(q2, alpha);
            q3 = vmulq_n_f32(q3, alpha);
            q0 = vmulq_f32(q0, q4);
            q1 = vmulq_f32(q1, q4);
            q2 = vmulq_f32(q2, q4);
            q3 = vmulq_f32(q3, q4);
            q0 = vmlaq_n_f32(q0, q8, beta);
            q1 = vmlaq_n_f32(q1, q9, beta);
            q2 = vmlaq_n_f32(q2, qA, beta);
            q3 = vmlaq_n_f32(q3, qB, beta);
            vst1q_f32(mat8 + j, q0);
            vst1q_f32(mat9 + j, q1);
            vst1q_f32(matA + j, q2);
            vst1q_f32(matB + j, q3);
        }
        for (; j < dimN; j++)
        {
            float a0 = mat8[j] * beta;
            float a1 = mat9[j] * beta;
            float a2 = matA[j] * beta;
            float a3 = matB[j] * beta;
            a0 += mat0[j] * mat4[j] * alpha;
            a1 += mat1[j] * mat4[j] * alpha;
            a2 += mat2[j] * mat4[j] * alpha;
            a3 += mat3[j] * mat4[j] * alpha;
            mat8[j] = a0;
            mat9[j] = a1;
            matA[j] = a2;
            matB[j] = a3;
        }
        mat0 += leadingN * 4;
        mat1 += leadingN * 4;
        mat2 += leadingN * 4;
        mat3 += leadingN * 4;
        mat8 += leadingN * 4;
        mat9 += leadingN * 4;
        matA += leadingN * 4;
        matB += leadingN * 4;
    }
    for (; i < dimM; i++)
    {
        int j = 0;
        for (j = 0; j <= dimN - 4; j += 4)
        {
            float32x4_t q0 = vld1q_f32(mat0 + j);
            float32x4_t q4 = vld1q_f32(mat4 + j);
            float32x4_t q8 = vld1q_f32(mat8 + j);
            q0 = vmulq_n_f32(q0, alpha);
            q0 = vmulq_f32(q0, q4);
            q0 = vmlaq_n_f32(q0, q8, beta);
            vst1q_f32(mat8 + j, q0);
        }
        for (; j < dimN; j++)
        {
            float a0 = mat0[j] * mat4[j] * alpha + mat8[j] * beta;
            mat8[j] = a0;
        }
        mat0 += leadingN;
        mat8 += leadingN;
    }
}

/**
 * @brief   vector_mul_matrix.
 *
 * @param   src1[in]     the input  vector(1*k)
 *          src2[in]     the input  matrix(k*n)
 *          dst[out]     the output vector(1*n)
 *          kn[in]       DIM_K & DIM_N
 *
 * @return  void
 */
void neon_vectormulmatrix_float(float * dst,
                                const float * src1,
                                const float * src2,
                                int *kn)
{
    int j ,l;
    int k = kn[0];
    int n = kn[1];
    const float * src1_p = src1;
    const float * src2_p = src2;
    float * dst_p = dst;
    for (j = 0; j <= n - 4; j += 4) {
        float32x2_t d16 = {0};
        float32x2_t d17 = {0};
        float32x2_t d18 = {0};
        float32x2_t d19 = {0};
        float32x2_t d20;
        float32x2_t d21;
        float32x4_t q0;
        src1_p = src1;
        src2_p = src2 + j * k;
        for (l = 0; l <= k - 4; l += 4) {
            // Matrix A
            float32x4_t q8  = vld1q_f32(src1_p);
            float32x2_t d0 = vget_low_f32(q8);
            float32x2_t d1 = vget_high_f32(q8);
            // Matrix B
            float32x4_t q12 = vld1q_f32(src2_p);
            float32x4_t q13 = vld1q_f32(src2_p + k);
            float32x4_t q14 = vld1q_f32(src2_p + k * 2);
            float32x4_t q15 = vld1q_f32(src2_p + k * 3);
            float32x2_t d8  = vget_low_f32(q12);
            float32x2_t d9  = vget_high_f32(q12);
            float32x2_t d10 = vget_low_f32(q13);
            float32x2_t d11 = vget_high_f32(q13);
            float32x2_t d12 = vget_low_f32(q14);
            float32x2_t d13 = vget_high_f32(q14);
            float32x2_t d14 = vget_low_f32(q15);
            float32x2_t d15 = vget_high_f32(q15);
            d16 = vmla_f32(d16, d0, d8);
            d17 = vmla_f32(d17, d0, d10);
            d18 = vmla_f32(d18, d0, d12);
            d19 = vmla_f32(d19, d0, d14);
            d16 = vmla_f32(d16, d1, d9);
            d17 = vmla_f32(d17, d1, d11);
            d18 = vmla_f32(d18, d1, d13);
            d19 = vmla_f32(d19, d1, d15);
            src1_p += 4;
            src2_p += 4;
        }// end for l
        d16 = vpadd_f32(d16, d17);
        d18 = vpadd_f32(d18, d19);
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for(; l < k; l ++) {
            float src1_d;
            src1_d = *src1_p;
            sum0 +=  src1_d * *src2_p;
            sum1 +=  src1_d * *(src2_p + k);
            sum2 +=  src1_d * *(src2_p + 2 * k);
            sum3 +=  src1_d * *(src2_p + 3 * k);
            src1_p++;
            src2_p++;
        }
        d20 = vset_lane_f32(sum0, d20, 0);
        d20 = vset_lane_f32(sum1, d20, 1);
        d21 = vset_lane_f32(sum2, d21, 0);
        d21 = vset_lane_f32(sum3, d21, 1);
        q0 = vaddq_f32(vcombine_f32(d16, d18), vcombine_f32(d20, d21));
        vst1q_f32(dst_p, q0);
        dst_p  += 4;
    }// end for j
}
/**
 * @brief   vector_dot_vector.
 *
 * @param   dst[out]     the output element(1*1)
 * @param   src1[in]     the input  vector(1*n)
 *          src2[in]     the input  vector(1*n)
 *          dimN[in]     size of vector
 *
 * @return  void
 */
void neon_VecdotVec(float *dst,
                    const float *src1,
                    const float *src2,
                    const int dimN)
{
    float *mat0 = (float *)src1;
    float *mat1 = (float *)src2;
    float32x4_t q0 = vld1q_f32(mat0);
    float32x4_t q1 = vld1q_f32(mat1);
    q0 = vmulq_f32(q0, q1);
    int j = 4;
    for (; j <= dimN - 4; j += 4)
    {
        float32x4_t q2 = vld1q_f32(mat0 + j);
        float32x4_t q3 = vld1q_f32(mat1 + j);
        q0 = vmlaq_f32(q0, q2, q3);
    }
    float32x2_t d0 = vpadd_f32(vget_low_f32(q0), vget_high_f32(q0));
    d0 = vpadd_f32(d0, d0);
    *dst = *((float *)&d0);
    for (; j < dimN; j++) {
        *dst += src1[j] * src2[j];
    }
}


