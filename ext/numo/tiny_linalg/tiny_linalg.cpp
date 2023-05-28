#include "tiny_linalg.hpp"
#include "dot.hpp"
#include "gesdd.hpp"
#include "gesvd.hpp"

VALUE rb_mTinyLinalg;
VALUE rb_mTinyLinalgBlas;
VALUE rb_mTinyLinalgLapack;

extern "C" void Init_tiny_linalg(void) {
  rb_require("numo/narray");

  rb_mTinyLinalg = rb_define_module_under(rb_mNumo, "TinyLinalg");
  rb_mTinyLinalgBlas = rb_define_module_under(rb_mTinyLinalg, "Blas");
  rb_mTinyLinalgLapack = rb_define_module_under(rb_mTinyLinalg, "Lapack");

  TinyLinalg::Dot<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DDot>::define_module_function(rb_mTinyLinalgBlas, "ddot");
  TinyLinalg::Dot<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SDot>::define_module_function(rb_mTinyLinalgBlas, "sdot");
  TinyLinalg::GESVD<TinyLinalg::numo_cDFloatId, TinyLinalg::numo_cDFloatId, double, double, TinyLinalg::DGESVD>::define_module_function(rb_mTinyLinalgLapack, "dgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cSFloatId, TinyLinalg::numo_cSFloatId, float, float, TinyLinalg::SGESVD>::define_module_function(rb_mTinyLinalgLapack, "sgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZGESVD>::define_module_function(rb_mTinyLinalgLapack, "zgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CGESVD>::define_module_function(rb_mTinyLinalgLapack, "cgesvd");
  TinyLinalg::GESDD<TinyLinalg::numo_cDFloatId, TinyLinalg::numo_cDFloatId, double, double, TinyLinalg::DGESDD>::define_module_function(rb_mTinyLinalgLapack, "dgesdd");
  TinyLinalg::GESDD<TinyLinalg::numo_cSFloatId, TinyLinalg::numo_cSFloatId, float, float, TinyLinalg::SGESDD>::define_module_function(rb_mTinyLinalgLapack, "sgesdd");
  TinyLinalg::GESDD<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZGESDD>::define_module_function(rb_mTinyLinalgLapack, "zgesdd");
  TinyLinalg::GESDD<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CGESDD>::define_module_function(rb_mTinyLinalgLapack, "cgesdd");
}
