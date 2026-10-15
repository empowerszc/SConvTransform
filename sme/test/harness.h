#ifndef SME_TEST_HARNESS_H
#define SME_TEST_HARNESS_H
#include "../runtime/sme_abi.h"
#include <math.h>
static inline int close_enough(float got, float want){
  float d = fabsf(got-want);
  return d <= 1e-3f*(1.0f+fabsf(want));
}
#endif
