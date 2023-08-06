#ifndef NUMO_TINY_LINALG_H
#define NUMO_TINY_LINALG_H 1

#include <cblas.h>
#include <lapacke.h>

#include <string>

#include <ruby.h>

#include <numo/narray.h>
#include <numo/template.h>

namespace TinyLinalg {

const VALUE NaryTypes[4] = {
  numo_cDFloat,
  numo_cSFloat,
  numo_cDComplex,
  numo_cSComplex
};

enum NaryType {
  numo_cDFloatId,
  numo_cSFloatId,
  numo_cDComplexId,
  numo_cSComplexId
};

} // namespace TinyLinalg

#endif /* NUMO_TINY_LINALG_H */
