#ifndef SME_ABI_H
#define SME_ABI_H
#include <stdint.h>
/* MLIR memref 描述符降级后的扁平 ABI：rank-1 = 5 个标量，rank-2 = 7 个标量。 */
typedef struct { float *allocated, *aligned; int64_t offset, size, stride; } MemRef1D;
typedef struct { float *allocated, *aligned; int64_t offset, size0, size1, stride0, stride1; } MemRef2D;
static inline MemRef1D mk1d(float *p, int64_t n){ MemRef1D m={p,p,0,n,1}; return m; }
static inline MemRef2D mk2d(float *p, int64_t r, int64_t c){
  MemRef2D m={p,p,0,r,c,c,1}; return m;
}
/* 计算 kernel（预 pad 输入/输出 + 任务区间 [T0,TN)） */
extern void sconv_sme_compute(
    int64_t N,int64_t Ic,int64_t IwP,int64_t OcP,int64_t OhP,int64_t OwP,
    int64_t T0,int64_t TN,
    float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t,
    float*,float*,int64_t,int64_t,int64_t,
    float*,float*,int64_t,int64_t,int64_t,int64_t,int64_t);
/* 对外入口（pad + 打包 + 线程 + unpad） */
int sconv_sme_direct(int64_t N,int64_t Ic,int64_t Ih,int64_t Iw,
                     int64_t Oc,int64_t Oh,int64_t Ow,
                     const float* in,const float* filter,float* out);
#endif
