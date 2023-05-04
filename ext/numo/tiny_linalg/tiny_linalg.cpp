#include "tiny_linalg.hpp"
#include "gesvd.hpp"

VALUE rb_mTinyLinalg;
VALUE rb_mTinyLinalgLapack;

extern "C" void Init_tiny_linalg(void) {
  rb_require("numo/narray");

  rb_mTinyLinalg = rb_define_module_under(rb_mNumo, "TinyLinalg");
  rb_mTinyLinalgLapack = rb_define_module_under(rb_mTinyLinalg, "Lapack");

  TinyLinalg::GESVD<TinyLinalg::numo_cDFloatId, TinyLinalg::numo_cDFloatId, double, double, TinyLinalg::DGESVD>::define_module_function(rb_mTinyLinalgLapack, "dgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cSFloatId, TinyLinalg::numo_cSFloatId, float, float, TinyLinalg::SGESVD>::define_module_function(rb_mTinyLinalgLapack, "sgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZGESVD>::define_module_function(rb_mTinyLinalgLapack, "zgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CGESVD>::define_module_function(rb_mTinyLinalgLapack, "cgesvd");
}
