//
//  cblas_hppl.h
//  demoapp
//
//  Created by shenjiangjiang on 15/10/15.
//  Copyright (c) 2015年 baidu. All rights reserved.
//

#ifndef __demoapp__cblas_hppl__
#define __demoapp__cblas_hppl__

#include <stdio.h>
#include <arm_neon.h>


typedef float       Elem_t;

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
                const int elemCnt);

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
void neon_matrixmul_4x8_c8_i32(int32_t * dst,
                               int8_t * src1,
                               int8_t * src2,
                               int *mkn);

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
                                int *mkn);

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
                             int *mkn);

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
                                int *mkn);

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
                 const int leadingN);

void print_shen(void);

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
                    const int leadingN);

/**
 * @brief   vector4 log.
 *
 * @param   src[in]     the input float32x4_t vector.
 *
 * @return  the output float32x4_t vector
 */
float32x4_t log_ps(float32x4_t src);

/**
 * @brief   vector4 exp.
 *
 * @param   src[in]     the input float32x4_t vector.
 *
 * @return  the output float32x4_t vector
 */
float32x4_t exp_ps(float32x4_t src);

/**
 * @brief   vector4 tanh.
 *
 * @param   src[in]     the input float32x4_t vector.
 *
 * @return  the output float32x4_t vector
 */
float32x4_t tanh_ps(float32x4_t src);

/**
 * @brief   sigmoid activation.
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *
 * @return  void
 */
void neon_sigmoid(const float *src,
                  float *dst,
                  const int dimM,
                  const int dimN,
                  const int leadingN);

/**
 * @brief   softmax activation.
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *
 * @return  void
 */
void neon_softmax(const float *src,
                  float *dst,
                  const int dimM,
                  const int dimN,
                  const int leadingN);


/**
 * @brief   relu activation.
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *
 * @return  void
 */
void neon_relu(const float *src,
               float *dst,
               const int dimM,
               const int dimN,
               const int leadingN);

/**
 * @brief   matrix log.
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *
 * @return  void
 */
void neon_log(const float *src,
              float *dst,
              const int dimM,
              const int dimN,
              const int leadingN);

/**
 * @brief   matrix tanh.
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *
 * @return  void
 */
void neon_tanh(const float *src,
               float *dst,
               const int dimM,
               const int dimN,
               const int leadingN);

/**
 * @brief   tansform float into char by scale
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *          scale transform scale of every row
 * @return  success HPPL_RET_OK.
 *          fail    HPPL_RET_ERROR.
 */

void neon_f2CRow(const float *src,
                 int8_t *dst,
                 const int DIM_M,
                 const int DIM_N,
                 const int leadingN,
                 float *scale);

/**
 * @brief   tansform float into unsigned char by scale
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *          scale transform scale of every row
 * @return  success HPPL_RET_OK.
 *          fail    HPPL_RET_ERROR.
 */

void neon_f2UC(const float *src,
               uint8_t *dst,
               const int DIM_M,
               const int DIM_N,
               const int leadingN,
               float *scale);


/**
 * @brief   tansform interge into float char by scale
 *          dst[i][j] = src[i][j] * alpha * scale1[i] * scale2[j] + dst[i][j] * beta
 *
 * @param   src[in]     the input matrix(m*n).
 *          dst[out]    the output matrix(m*n).
 *          dimM[in]    row number.
 *          dimN[in]    column number.
 *          leadingN[in]the aligned column number
 *          scale1 transform scale of column
 *          scale2 transform scale of row
 *          alpha ratio of src
 *          beta  ratio of dst
 * @return  success HPPL_RET_OK.
 *          fail    HPPL_RET_ERROR.
 */


void neon_i2F(const int *src,
              float *dst,
              const int DIM_M,
              const int DIM_N,
              const int leadingN,
              const float *scale1,
              const float *scale2,
              const float alpha,
              const float beta);

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
                                int *kn);

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
                                int *kn);

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
void neon_vectormulmatrix_char(int32_t *dst,
                               int8_t *src1,
                               int8_t *src2,
                               int *kn);

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
                    const int dimN);

/**
 * @brief   neon_vector_i2F
 *
 * @param   src_data[in]     the input  vector(1*n)
 *          dst_data[out]    the output vector(1*n)
 *          DIM_N[in]        n
 *          scaleA[in]       vector scale factor
 *          scaleB[in]       matrix scale factor
 *
 * @return  void
 */
void neon_vector_i2F(int32_t *src_data,
                     float *dst_data,
                     int DIM_N,
                     float scaleA,
                     float *scaleB);


void neon_matrixmul_4x8_c8_i32(int32_t * dst,
                               int8_t * src1,
                               int8_t * src2,
                               int *mkn);

void neon_matrixmul_4x8_uc8_i32(int32_t * dst,
                                uint8_t * src1,
                                int8_t * src2,
                                int *mkn);


#endif /* defined(__demoapp__cblas_hppl__) */
