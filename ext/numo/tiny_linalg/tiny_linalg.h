#ifndef NUMO_TINY_LINALG_H
#define NUMO_TINY_LINALG_H 1

#if defined(TINYLINALG_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <ruby.h>

#endif /* NUMO_TINY_LINALG_H */
