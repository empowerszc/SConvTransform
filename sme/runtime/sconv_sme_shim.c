#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "sme_abi.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>

#define R 16   /* SVL=512 -> R=16（本机已确认） */

typedef struct {
  int64_t N,Ic,IwP,OcP,OhP,OwP,T0,TN;
  float *in,*wpack,*out;
} Job;

static void run_range(Job*j){
  MemRef2D mi = mk2d(j->in, j->N*j->Ic, j->IwP*(j->OhP+2));
  MemRef1D mw = mk1d(j->wpack, j->Ic*9*j->OcP);
  MemRef2D mo = mk2d(j->out, j->N*j->OcP, j->OhP*j->OwP);
  sconv_sme_compute(j->N,j->Ic,j->IwP,j->OcP,j->OhP,j->OwP,j->T0,j->TN,
    mi.allocated,mi.aligned,mi.offset,mi.size0,mi.size1,mi.stride0,mi.stride1,
    mw.allocated,mw.aligned,mw.offset,mw.size,mw.stride,
    mo.allocated,mo.aligned,mo.offset,mo.size0,mo.size1,mo.stride0,mo.stride1);
}
static void*worker(void*p){ run_range((Job*)p); return NULL; }

static int sme_threads(void){
  const char*e=getenv("SCONV_SME_THREADS");
  if(e&&*e){int n=atoi(e); if(n>0)return n;}
  /* Respect the process CPU affinity (taskset/numactl/cgroups) so we do not
     oversubscribe the cores actually available to us. */
  cpu_set_t set;
  if(sched_getaffinity(0,sizeof(set),&set)==0){
    int n=CPU_COUNT(&set);
    if(n>0)return n;
  }
  long n=sysconf(_SC_NPROCESSORS_ONLN);
  return n>0?(int)n:1;
}

int sconv_sme_direct(int64_t N,int64_t Ic,int64_t Ih,int64_t Iw,
                     int64_t Oc,int64_t Oh,int64_t Ow,
                     const float*in,const float*filter,float*out){
  int64_t OhP=(Oh+3)/4*4, OwP=(Ow+R-1)/R*R, OcP=(Oc+R-1)/R*R;
  int64_t IhP=OhP+2, IwP=OwP+2;

  /* 权重打包：wpack[(ic*9+kh*3+kw)*OcP + oc] */
  float*wpack=calloc((size_t)Ic*9*OcP,sizeof(float));
  for(int64_t ic=0;ic<Ic;ic++)
    for(int64_t kh=0;kh<3;kh++)
      for(int64_t kw=0;kw<3;kw++)
        for(int64_t oc=0;oc<Oc;oc++)
          wpack[(ic*9+kh*3+kw)*OcP+oc]=filter[((oc*Ic+ic)*3+kh)*3+kw];

  /* 输入：框架已左上预 pad；仅在尾块需要时把行/列扩到 IhP×IwP（右下补零）。 */
  float*pin; int need_pad = (Ih!=IhP)||(Iw!=IwP);
  if(!need_pad){ pin=(float*)in; }
  else {
    pin=calloc((size_t)N*Ic*IhP*IwP,sizeof(float));
    for(int64_t n=0;n<N;n++)
      for(int64_t ic=0;ic<Ic;ic++)
        for(int64_t r=0;r<Ih;r++)
          memcpy(pin+((n*Ic+ic)*IhP+r)*IwP,
                 in +((n*Ic+ic)*Ih +r)*Iw,
                 (size_t)Iw*sizeof(float));
  }

  /* 预 pad 输出 */
  float*pout=calloc((size_t)N*OcP*OhP*OwP,sizeof(float));

  int64_t Ocb=OcP/R, OhB=OhP/4, T=N*Ocb*OhB;
  int nt=sme_threads(); if(nt>T)nt=(int)T; if(nt<1)nt=1;

  Job base={N,Ic,IwP,OcP,OhP,OwP,0,0,pin,wpack,pout};
  if(nt==1){ base.T0=0; base.TN=T; run_range(&base); }
  else {
    pthread_t*th=malloc(sizeof(pthread_t)*nt);
    Job*ar=malloc(sizeof(Job)*nt);
    int64_t per=(T+nt-1)/nt, t=0;
    for(int i=0;i<nt;i++){
      int64_t cnt=per; if(t+cnt>T)cnt=T-t; if(cnt<0)cnt=0;
      ar[i]=base; ar[i].T0=t; ar[i].TN=cnt; t+=cnt;
      pthread_create(&th[i],NULL,worker,&ar[i]);
    }
    for(int i=0;i<nt;i++)pthread_join(th[i],NULL);
    free(th); free(ar);
  }

  /* unpad 输出（只回写有效 oc<Oc,oh<Oh,ow<Ow） */
  for(int64_t n=0;n<N;n++)
    for(int64_t oc=0;oc<Oc;oc++)
      for(int64_t oh=0;oh<Oh;oh++)
        memcpy(out+((n*Oc+oc)*Oh+oh)*Ow,
               pout+((n*OcP+oc)*OhP+oh)*OwP,
               (size_t)Ow*sizeof(float));

  if(need_pad)free(pin);
  free(wpack); free(pout);
  return 0;
}
