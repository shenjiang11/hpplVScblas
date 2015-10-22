//
//  ViewController.m
//  demoapp
//
//  Created by shenjiangjiang on 15/7/21.
//  Copyright (c) 2015年 baidu. All rights reserved.
//
#import <unistd.h>
#import <stdlib.h>
#import <math.h>
#import <sys/time.h>
#import <Accelerate/Accelerate.h>

#import "ViewController.h"
#include "cblas_hppl.h"

//#define SGEMM 1
#define SGEMV 1
//#define CGEMM 1
//#define CGEMV 1



#ifdef CGEMM
#define SRC1_TYPE int8_t
#define SRC2_TYPE int8_t
#define DEST_TYPE int32_t
#define CBLAS_FUNC "cblas_cgemm"
#define HPPL_FUNC  "hppl_cgemm "
#endif

#ifdef CGEMV
#define SRC1_TYPE int8_t
#define SRC2_TYPE int8_t
#define DEST_TYPE int32_t
#define CBLAS_FUNC "cblas_cgemv"
#define HPPL_FUNC  "hppl_cgemv "
#endif

#ifdef SGEMM
#define SRC1_TYPE float
#define SRC2_TYPE float
#define DEST_TYPE float
#define CBLAS_FUNC "cblas_cgemm"
#define HPPL_FUNC  "hppl_cgemm "
#endif

#ifdef SGEMV
#define SRC1_TYPE float
#define SRC2_TYPE float
#define DEST_TYPE float
#define CBLAS_FUNC "cblas_sgemv"
#define HPPL_FUNC  "hppl_sgemv "
#endif

#define TEST_LOOP 10

@interface ViewController ()
{
    SRC1_TYPE *src1;
    SRC2_TYPE *src2;
    DEST_TYPE *dest_cblas;
    DEST_TYPE *dest_hppl;
    
    unsigned long time_before;
    unsigned long time_after;
    
    unsigned long time_cblas_max;
    unsigned long time_cblas_min;
    float time_cblas_avg;
    
    unsigned long time_hppl_max;
    unsigned long time_hppl_min;
    float time_hppl_avg;
    
    /* specify transpose for matrix */
    enum CBLAS_TRANSPOSE transA;
    enum CBLAS_TRANSPOSE transB;
    
    int dimm;
    int dimn;
    int dimk;
    
    unsigned long long matrix_scale;
}
@end

@implementation ViewController

void randomize(SRC1_TYPE *buf, int size)
{
    int i    = 0;
    int temp = 0;
    
    /* init rand */
    srand((unsigned)time(NULL));
    
    /* init weight */
    for(i = 0; i < size; i++)
    {
        temp = rand();
        *buf = ((temp > RAND_MAX/2)?1:-1)*sqrt(abs(temp-RAND_MAX/2)/(SRC1_TYPE)RAND_MAX);
        buf++;
    }
}

unsigned long timestamp_gettimeofday()
{
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return 1000000ULL * (unsigned long)( tv.tv_sec ) + (unsigned long)( tv.tv_usec );
}

/*
 void print_matrix(float *matrix)
 {
 int i, j;
 float *matrix_int = matrix;
 printf("\n");
 for(i=0; i<dimn; i++)
 {
 for(j=0; j<dimm; j++)
 printf("%1.1f ", matrix_int[i*DIM_N + j]);
 printf("\n");
 }
 printf("\n");
 }*/

- (int) setup_buffer
{
    src1 = (SRC1_TYPE *)malloc(dimm*dimn*sizeof(SRC1_TYPE));
    src2 = (SRC2_TYPE *)malloc(dimn*dimk*sizeof(SRC2_TYPE));
    dest_cblas = (DEST_TYPE *)malloc(dimm*dimk*sizeof(DEST_TYPE));
    dest_hppl = (DEST_TYPE *)malloc(dimm*dimk*sizeof(DEST_TYPE));
    
    if (!src1 || !src2 || !dest_cblas || !dest_hppl) {
        printf("malloc failed\n");
        return -1;
    }
    
    randomize(src1, dimm*dimn);
    randomize(src2, dimn*dimk);
    memset(dest_cblas, 0, dimm*dimk*sizeof(DEST_TYPE));
    memset(dest_hppl, 0, dimm*dimk*sizeof(DEST_TYPE));
    transA = CblasNoTrans;
    transB = CblasTrans;
    
    matrix_scale = (double)(1.0e-3 * (2.0*dimm * dimn * dimk + dimm*dimn)); /*in gflops*/
#ifdef SGEMV
    cblas_sgemv(CblasRowMajor, CblasNoTrans, dimm, dimn, 1.0, src1, dimn, src2, 1.0, 0, dest_cblas, 1);
#endif
    return 0;
}

- (void) test_clbas
{
    int i;
    unsigned long time_tmp;
    
    time_before = timestamp_gettimeofday();
    
    for(i = 0; i < TEST_LOOP; i++) {
#ifdef SGEMM
        cblas_sgemm(CblasRowMajor, transA, transB, dimm, dimk, dimn,
                    1.0, src1, dimn, src2, dimk, 0.0, dest_cblas, dimk);
#endif
        
#ifdef SGEMV
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dimm, dimn, 1.0, src1, dimn, src2, 1.0, 0, dest_cblas, 1);
#endif
        
#ifdef CGEMM
        int alpha = 1;
        int beta = 0;
        
        cblas_cgemm(CblasRowMajor, transA, transB, dimm, dimk, dimn,
                    &alpha, src1, dimn, src2, dimk, &beta, dest_cblas, dimk);
#endif
        
#ifdef CGEMV
        int alpha = 1;
        int beta = 0;
        
        cblas_cgemv(CblasRowMajor, CblasNoTrans, dimm, dimn, &alpha, src1, dimn, src2, 1, &beta, dest_cblas, 1);
#endif
    }
    
    time_after = timestamp_gettimeofday();
    
    time_tmp = (time_after - time_before);
    time_cblas_avg += time_tmp;
    if (time_cblas_min == 0)
        time_cblas_min = time_cblas_max = time_tmp;
    if (time_cblas_max < time_tmp)
        time_cblas_max = time_tmp;
    if (time_cblas_min > time_tmp)
        time_cblas_min = time_tmp;
}

- (void) test_hppl
{
    int i;
    unsigned long time_tmp;
    
    time_before = timestamp_gettimeofday();
    
    for(i = 0; i < TEST_LOOP; i++) {
#ifdef SGEMM
        int dims[3] = {dimm, dimn, dimk};
        neon_matrixmul_4x4float(dest_hppl, src1, src2, dims);
#endif
        
#ifdef SGEMV
        int dims[2] = {dimn, dimm};
        neon_vectormulmatrix_float(dest_hppl, src2, src1, dims);
#endif
        
#ifdef CGEMM
        int dims[3] = {dimm, dimn, dimk};
        neon_matrixmul_4x8_c8_i32(dest_hppl, src1, src2, dims);
#endif
        
#ifdef CGEMV
        int dims[2] = {dimm, dimn};
        neon_vectormulmatrix_char(dest_hppl, src2, src1, dims);
#endif
    }
    
    time_after = timestamp_gettimeofday();
    
    time_tmp = (time_after - time_before);
    time_hppl_avg += time_tmp;
    if (time_hppl_min == 0)
        time_hppl_min = time_hppl_max = time_tmp;
    if (time_hppl_max < time_tmp)
        time_hppl_max = time_tmp;
    if (time_hppl_min > time_tmp)
        time_hppl_min = time_tmp;
}

- (void) cal_result
{
    BOOL test_pass = true;
    
    for(int i = 0; i < (dimm*dimk); i++)
    {
        DEST_TYPE a = dest_cblas[i];
        DEST_TYPE b = dest_hppl[i];
        if (fabs(a - b) > 0.2)
        {
            printf("\nresult diff, i = %d, the abs is %d\n", i, fabs(a - b));
            printf("dest_cblas[i] = %f, dest_hppl[i] = %f\n", a, b);
            test_pass = false;
            break;
        }
    }
    
    if (!test_pass)
        printf("test fail!\n\n");
}

- (void) print_result
{
    time_cblas_avg /= TEST_LOOP;
    time_hppl_avg /= TEST_LOOP;
    
    printf("\nwhen m = %d, n = %d, k =%d\n", dimm, dimn, dimk);
    printf("Function     Avg Gflops/s  Avg time     Min time     Max time\n");
    printf("%s%11.2f  %10.2f  %11lu %11lu\n", CBLAS_FUNC,
           matrix_scale/(float)time_cblas_avg, time_cblas_avg,
           time_cblas_min, time_cblas_max);
    printf("%s%11.2f  %10.2f  %11lu %11lu\n", HPPL_FUNC,
           matrix_scale/(float)time_hppl_avg, time_hppl_avg,
           time_hppl_min, time_hppl_max);
}

- (void) free_mem
{
    if (src1)
        free(src1);
    if (src2)
        free(src2);
    if (dest_cblas)
        free(dest_cblas);
    if (dest_hppl)
        free(dest_hppl);
}

- (void) test_main
{
    int i;
    
    time_cblas_min = time_cblas_avg = time_hppl_max = 0;
    time_hppl_avg = time_hppl_min = time_hppl_max = 0;
    
    [self setup_buffer];
    [self test_clbas];
    [self test_hppl];
    [self cal_result];
    [self free_mem];
    
    
    [self print_result];
}


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    for(int i = 64; i <= 2048; i *=2) {
        dimm = dimn = i;
#ifdef CGEMM
        dimk = i;
#endif
        
#ifdef SGEMM
        dimk = i;
#endif
        
#ifdef CGEMV
        dimk = 1;
#endif
        
#ifdef SGEMV
        dimk = 1;
#endif
        
        [self test_main];
    }
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
