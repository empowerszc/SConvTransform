#include "harness.h"
#include <stdio.h>
#include <stdlib.h>

/* 输入为“已预 pad”的 NCHW：Ih=Oh+2, Iw=Ow+2（valid conv）。 */
static int run(int N,int Ic,int Oh,int Ow,int Oc){
  int Ih=Oh+2, Iw=Ow+2;
  float*in=malloc(sizeof(float)*(size_t)N*Ic*Ih*Iw);
  float*flt=malloc(sizeof(float)*(size_t)Oc*Ic*9);
  float*out=malloc(sizeof(float)*(size_t)N*Oc*Oh*Ow);
  float*ref=malloc(sizeof(float)*(size_t)N*Oc*Oh*Ow);
  for(size_t i=0;i<(size_t)N*Ic*Ih*Iw;i++) in[i]=0.1f*(float)((i%29)-13);
  for(size_t i=0;i<(size_t)Oc*Ic*9;i++)    flt[i]=0.2f*(float)((i%17)-8);
  for(int n=0;n<N;n++)for(int oc=0;oc<Oc;oc++)for(int oh=0;oh<Oh;oh++)for(int ow=0;ow<Ow;ow++){
    float s=0;
    for(int ic=0;ic<Ic;ic++)for(int kh=0;kh<3;kh++)for(int kw=0;kw<3;kw++)
      s+=in[((size_t)(n*Ic+ic)*Ih+oh+kh)*Iw+ow+kw]*flt[(((size_t)oc*Ic+ic)*3+kh)*3+kw];
    ref[((size_t)(n*Oc+oc)*Oh+oh)*Ow+ow]=s;
  }
  sconv_sme_direct(N,Ic,Ih,Iw,Oc,Oh,Ow,in,flt,out);
  int bad=0;
  for(size_t i=0;i<(size_t)N*Oc*Oh*Ow;i++) if(!close_enough(out[i],ref[i])) bad++;
  printf("N%d Ic%d Oh%d Ow%d Oc%d: %s (%d bad)\n",N,Ic,Oh,Ow,Oc,bad?"FAIL":"PASS",bad);
  free(in);free(flt);free(out);free(ref);
  return bad?1:0;
}
/* The public entry must reject a geometry that the kernel cannot handle
   (here `Ih > Oh+2`, which would over-read the input) rather than return
   success. */
static int run_invalid(void){
  int N=1,Ic=2,Ih=12,Iw=12,Oc=16,Oh=8,Ow=8;
  float*in=malloc(sizeof(float)*(size_t)N*Ic*Ih*Iw);
  float*flt=malloc(sizeof(float)*(size_t)Oc*Ic*9);
  float*out=malloc(sizeof(float)*(size_t)N*Oc*Oh*Ow);
  int r=sconv_sme_direct(N,Ic,Ih,Iw,Oc,Oh,Ow,in,flt,out);
  printf("invalid geometry rejected: %s (ret=%d)\n", r?"OK":"FAIL", r);
  free(in);free(flt);free(out);
  return r?0:1;
}
int main(void){
  int bad=0;
  bad|=run(1,2,8,8,16);
  bad|=run(1,3,10,10,17);
  bad|=run(2,5,7,13,32);
  bad|=run_invalid();
  printf(bad?"SHIM FAIL\n":"SHIM PASS\n");
  return bad?1:0;
}
