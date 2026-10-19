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

/* ---- persistent worker pool (created once per thread, reused every call) ---
   Creating/throwing away a team of `threads` per call cost ~2 ms/call on this
   machine, which dominated short shapes. Keep a broadcast/generation pool and
   run two phases per call (pack+pad, then compute) without re-spawning. */
typedef struct Pool {
  pthread_mutex_t mtx;
  pthread_cond_t  cv;
  int nworkers, go, done, stop;
  void (*task)(void *, int);
  void *ctx;
  pthread_t *th;
  struct WArg { struct Pool *p; int id; } *wa;
} Pool;

static void *pool_worker(void *a) {
  struct WArg *w = a;
  Pool *p = w->p;
  int seen = 0;
  pthread_mutex_lock(&p->mtx);
  for (;;) {
    while (p->go == seen && !p->stop) pthread_cond_wait(&p->cv, &p->mtx);
    if (p->stop) break;
    seen = p->go;
    void (*task)(void *, int) = p->task;
    void *ctx = p->ctx;
    pthread_mutex_unlock(&p->mtx);
    task(ctx, w->id);
    pthread_mutex_lock(&p->mtx);
    p->done++;
    pthread_cond_broadcast(&p->cv);   /* one CV serves both go/done: broadcast */
  }
  pthread_mutex_unlock(&p->mtx);
  return NULL;
}

static int sme_threads(void) {
  const char *e = getenv("SCONV_SME_THREADS");
  if (e && *e) { int n = atoi(e); if (n > 0) return n; }
  /* Respect the process CPU affinity (taskset/numactl/cgroups) so we do not
     oversubscribe the cores actually available to us. */
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    int n = CPU_COUNT(&set);
    if (n > 0) return n;
  }
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
}

static Pool *pool_get(int nt) {
  static _Thread_local Pool *pl = NULL;
  if (pl && pl->nworkers == nt - 1) return pl;
  if (pl) { /* resize: stop the old team */
    pthread_mutex_lock(&pl->mtx);
    pl->stop = 1;
    pthread_cond_broadcast(&pl->cv);
    pthread_mutex_unlock(&pl->mtx);
    for (int i = 0; i < pl->nworkers; i++) pthread_join(pl->th[i], NULL);
    pthread_mutex_destroy(&pl->mtx);
    pthread_cond_destroy(&pl->cv);
    free(pl->th); free(pl->wa); free(pl);
    pl = NULL;
  }
  pl = calloc(1, sizeof(Pool));
  if (!pl) return NULL;
  pthread_mutex_init(&pl->mtx, NULL);
  pthread_cond_init(&pl->cv, NULL);
  int want = nt - 1;
  pl->th = malloc(sizeof(pthread_t) * (want > 0 ? want : 1));
  pl->wa = malloc(sizeof(struct WArg) * (want > 0 ? want : 1));
  if (!pl->th || !pl->wa) { free(pl->th); free(pl->wa); free(pl); return NULL; }
  int made = 0;
  for (int i = 0; i < want; i++) {
    pl->wa[i].p = pl; pl->wa[i].id = i + 1;
    if (pthread_create(&pl->th[i], NULL, pool_worker, &pl->wa[i]) != 0) break;
    made++;
  }
  pl->nworkers = made;
  return pl;
}

static void pool_run(Pool *p, void (*task)(void *, int), void *ctx) {
  if (!p || p->nworkers == 0) { task(ctx, 0); return; }
  pthread_mutex_lock(&p->mtx);
  p->task = task; p->ctx = ctx; p->done = 0; p->go++;
  pthread_cond_broadcast(&p->cv);
  pthread_mutex_unlock(&p->mtx);
  task(ctx, 0);                 /* main thread is participant 0 */
  pthread_mutex_lock(&p->mtx);
  while (p->done < p->nworkers) pthread_cond_wait(&p->cv, &p->mtx);
  pthread_mutex_unlock(&p->mtx);
}

/* ---- persistent per-thread scratch (avoids per-call calloc + page faults) -- */
typedef struct { float *p; size_t cap; } Buf;
static _Thread_local Buf sc_wpack, sc_pin, sc_pout;
static float *grow(Buf *b, size_t n) {
  if (b->cap < n) { float *q = realloc(b->p, n * sizeof(float)); if (!q) return NULL; b->p = q; b->cap = n; }
  return b->p;
}

/* ---- per-call context + the two phases ---- */
typedef struct {
  int64_t N, Ic, Ih, Iw, IhP, IwP, Oc, OcP, OhP, OwP, T;
  const float *in, *filter;
  float *wpack, *pin, *pout;
  int need_pad, nt;             /* nt = participants (nworkers + 1) */
} Ctx;

/* phase A: weight pack (over Ic) + input pad (over N*Ic); disjoint outputs. */
static void task_packpad(void *c, int id) {
  Ctx *x = c;
  int nt = x->nt;
  int64_t per = (x->Ic + nt - 1) / nt, lo = (int64_t)id * per, hi = lo + per;
  if (hi > x->Ic) hi = x->Ic;
  for (int64_t ic = lo; ic < hi; ic++)
    for (int64_t kh = 0; kh < 3; kh++)
      for (int64_t kw = 0; kw < 3; kw++)
        for (int64_t oc = 0; oc < x->Oc; oc++)
          x->wpack[(ic * 9 + kh * 3 + kw) * x->OcP + oc] =
              x->filter[((oc * x->Ic + ic) * 3 + kh) * 3 + kw];

  if (!x->need_pad) return;
  int64_t tot = x->N * x->Ic, p2 = (tot + nt - 1) / nt, l2 = (int64_t)id * p2, h2 = l2 + p2;
  if (h2 > tot) h2 = tot;
  for (int64_t idx = l2; idx < h2; idx++) {
    float *dst = x->pin + idx * x->IhP * x->IwP;   /* layout is (n*Ic+ic)*IhP*IwP */
    memset(dst, 0, sizeof(float) * (size_t)(x->IhP * x->IwP));
    const float *src = x->in + idx * x->Ih * x->Iw;
    for (int64_t r = 0; r < x->Ih; r++)
      memcpy(dst + r * x->IwP, src + r * x->Iw, sizeof(float) * (size_t)x->Iw);
  }
}

/* phase B: compute tasks [T0,TN) for this participant. */
static void task_compute(void *c, int id) {
  Ctx *x = c;
  int nt = x->nt;
  int64_t per = (x->T + nt - 1) / nt, t0 = (int64_t)id * per, t1 = t0 + per;
  if (t1 > x->T) t1 = x->T;
  if (t0 > x->T) t0 = x->T;
  MemRef2D mi = mk2d(x->pin, x->N * x->Ic, x->IwP * (x->OhP + 2));
  MemRef1D mw = mk1d(x->wpack, x->Ic * 9 * x->OcP);
  MemRef2D mo = mk2d(x->pout, x->N * x->OcP, x->OhP * x->OwP);
  sconv_sme_compute(x->N, x->Ic, x->IwP, x->OcP, x->OhP, x->OwP, t0, t1 - t0,
    mi.allocated, mi.aligned, mi.offset, mi.size0, mi.size1, mi.stride0, mi.stride1,
    mw.allocated, mw.aligned, mw.offset, mw.size, mw.stride,
    mo.allocated, mo.aligned, mo.offset, mo.size0, mo.size1, mo.stride0, mo.stride1);
}

int sconv_sme_direct(int64_t N,int64_t Ic,int64_t Ih,int64_t Iw,
                     int64_t Oc,int64_t Oh,int64_t Ow,
                     const float*in,const float*filter,float*out){
  /* The kernel computes a valid conv: it needs Oh==Ih-2 rows / Ow==Iw-2 cols of
     input, and must not read more rows/cols than the padded buffers hold. */
  if (N<=0||Ic<=0||Oc<=0||Oh<=0||Ow<=0||Ih<Oh+2||Iw<Ow+2)
    return -1;

  int64_t OhP=(Oh+3)/4*4, OwP=(Ow+R-1)/R*R, OcP=(Oc+R-1)/R*R;
  int64_t IhP=OhP+2, IwP=OwP+2;
  if (Ih>IhP||Iw>IwP)
    return -1;

  int64_t Ocb=OcP/R, OhB=OhP/4, T=N*Ocb*OhB;
  int nt=sme_threads(); if(nt>T)nt=(int)T; if(nt<1)nt=1;

  /* Reuse capacity-sized scratch (kernel overwrites the whole padded output;
     wpack/pin are fully rewritten each call). */
  float*wpack=grow(&sc_wpack,(size_t)Ic*9*OcP);
  if(!wpack) return -1;
  int need_pad = (Ih!=IhP)||(Iw!=IwP);
  float*pin=(float*)in;
  if(need_pad){ pin=grow(&sc_pin,(size_t)N*Ic*IhP*IwP); if(!pin) return -1; }
  float*pout=grow(&sc_pout,(size_t)N*OcP*OhP*OwP);
  if(!pout) return -1;

  Ctx x={N,Ic,Ih,Iw,IhP,IwP,Oc,OcP,OhP,OwP,T,in,filter,wpack,pin,pout,need_pad,nt};

  if(nt==1){
    task_packpad(&x,0);
    task_compute(&x,0);
  } else {
    Pool*p=pool_get(nt);
    if(!p){ task_packpad(&x,0); task_compute(&x,0); }
    else {
      x.nt = p->nworkers + 1;      /* actual participants (create may have failed) */
      pool_run(p,task_packpad,&x);
      pool_run(p,task_compute,&x);
    }
  }

  /* unpad 输出（只回写有效 oc<Oc,oh<Oh,ow<Ow） */
  for(int64_t n=0;n<N;n++)
    for(int64_t oc=0;oc<Oc;oc++)
      for(int64_t oh=0;oh<Oh;oh++)
        memcpy(out+((n*Oc+oc)*Oh+oh)*Ow,
               pout+((n*OcP+oc)*OhP+oh)*OwP,
               (size_t)Ow*sizeof(float));

  return 0;
}
