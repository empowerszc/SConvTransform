#include <cblas.h>
/* SConv microkernel: C[N,M] += B[K,N]^T * A[K,M]
   A = input tile [K,M] row-major (lda = M)
   B = filter tile [K,N] row-major (ldb = N)
   C = output tile [N,M] row-major (ldc given)
   Equivalent: C = B^T * A + C */
int sgemm_blas_kernel(long m, long n, long k, float alpha,
                      float *A, float *B, float *C, long ldc) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, m, k, alpha, B, n, A, m, 1.0f, C, ldc);
    return 0;
}
