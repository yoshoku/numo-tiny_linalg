#include "tiny_linalg.hpp"
#include "converter.hpp"
#include "dot.hpp"
#include "dot_sub.hpp"
#include "gemm.hpp"
#include "gemv.hpp"
#include "gesdd.hpp"
#include "gesvd.hpp"
#include "nrm2.hpp"

VALUE rb_mTinyLinalg;
VALUE rb_mTinyLinalgBlas;
VALUE rb_mTinyLinalgLapack;

char blas_char(VALUE nary_arr) {
  char type = 'n';
  const size_t n = RARRAY_LEN(nary_arr);
  for (size_t i = 0; i < n; i++) {
    VALUE arg = rb_ary_entry(nary_arr, i);
    if (RB_TYPE_P(arg, T_ARRAY)) {
      arg = rb_funcall(numo_cNArray, rb_intern("asarray"), 1, arg);
    }
    if (CLASS_OF(arg) == numo_cBit || CLASS_OF(arg) == numo_cInt64 || CLASS_OF(arg) == numo_cInt32 ||
        CLASS_OF(arg) == numo_cInt16 || CLASS_OF(arg) == numo_cInt8 || CLASS_OF(arg) == numo_cUInt64 ||
        CLASS_OF(arg) == numo_cUInt32 || CLASS_OF(arg) == numo_cUInt16 || CLASS_OF(arg) == numo_cUInt8) {
      if (type == 'n') {
        type = 'd';
      }
    } else if (CLASS_OF(arg) == numo_cDFloat) {
      if (type == 'c' || type == 'z') {
        type = 'z';
      } else {
        type = 'd';
      }
    } else if (CLASS_OF(arg) == numo_cSFloat) {
      if (type == 'n') {
        type = 's';
      }
    } else if (CLASS_OF(arg) == numo_cDComplex) {
      type = 'z';
    } else if (CLASS_OF(arg) == numo_cSComplex) {
      if (type == 'n' || type == 's') {
        type = 'c';
      } else if (type == 'd') {
        type = 'z';
      }
    }
  }
  return type;
}

static VALUE tiny_linalg_blas_char(int argc, VALUE* argv, VALUE self) {
  VALUE nary_arr = Qnil;
  rb_scan_args(argc, argv, "*", &nary_arr);

  const char type = blas_char(nary_arr);
  if (type == 'n') {
    rb_raise(rb_eTypeError, "invalid data type for BLAS/LAPACK");
    return Qnil;
  }

  return rb_str_new(&type, 1);
}

extern "C" void Init_tiny_linalg(void) {
  rb_require("numo/narray");

  rb_mTinyLinalg = rb_define_module_under(rb_mNumo, "TinyLinalg");
  rb_mTinyLinalgBlas = rb_define_module_under(rb_mTinyLinalg, "Blas");
  rb_mTinyLinalgLapack = rb_define_module_under(rb_mTinyLinalg, "Lapack");

  rb_define_module_function(rb_mTinyLinalg, "blas_char", RUBY_METHOD_FUNC(tiny_linalg_blas_char), -1);

  TinyLinalg::Dot<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DDot>::define_module_function(rb_mTinyLinalgBlas, "ddot");
  TinyLinalg::Dot<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SDot>::define_module_function(rb_mTinyLinalgBlas, "sdot");
  TinyLinalg::DotSub<TinyLinalg::numo_cDComplexId, double, TinyLinalg::ZDotuSub>::define_module_function(rb_mTinyLinalgBlas, "zdotu");
  TinyLinalg::DotSub<TinyLinalg::numo_cSComplexId, float, TinyLinalg::CDotuSub>::define_module_function(rb_mTinyLinalgBlas, "cdotu");
  TinyLinalg::Gemm<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DGemm, TinyLinalg::DConverter>::define_module_function(rb_mTinyLinalgBlas, "dgemm");
  TinyLinalg::Gemm<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SGemm, TinyLinalg::SConverter>::define_module_function(rb_mTinyLinalgBlas, "sgemm");
  TinyLinalg::Gemm<TinyLinalg::numo_cDComplexId, dcomplex, TinyLinalg::ZGemm, TinyLinalg::ZConverter>::define_module_function(rb_mTinyLinalgBlas, "zgemm");
  TinyLinalg::Gemm<TinyLinalg::numo_cSComplexId, scomplex, TinyLinalg::CGemm, TinyLinalg::CConverter>::define_module_function(rb_mTinyLinalgBlas, "cgemm");
  TinyLinalg::Gemv<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DGemv, TinyLinalg::DConverter>::define_module_function(rb_mTinyLinalgBlas, "dgemv");
  TinyLinalg::Gemv<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SGemv, TinyLinalg::SConverter>::define_module_function(rb_mTinyLinalgBlas, "sgemv");
  TinyLinalg::Gemv<TinyLinalg::numo_cDComplexId, dcomplex, TinyLinalg::ZGemv, TinyLinalg::ZConverter>::define_module_function(rb_mTinyLinalgBlas, "zgemv");
  TinyLinalg::Gemv<TinyLinalg::numo_cSComplexId, scomplex, TinyLinalg::CGemv, TinyLinalg::CConverter>::define_module_function(rb_mTinyLinalgBlas, "cgemv");
  TinyLinalg::Nrm2<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DNrm2>::define_module_function(rb_mTinyLinalgBlas, "dnrm2");
  TinyLinalg::Nrm2<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SNrm2>::define_module_function(rb_mTinyLinalgBlas, "snrm2");
  TinyLinalg::Nrm2<TinyLinalg::numo_cDComplexId, double, TinyLinalg::DZNrm2>::define_module_function(rb_mTinyLinalgBlas, "dznrm2");
  TinyLinalg::Nrm2<TinyLinalg::numo_cSComplexId, float, TinyLinalg::SCNrm2>::define_module_function(rb_mTinyLinalgBlas, "scnrm2");
  TinyLinalg::GESVD<TinyLinalg::numo_cDFloatId, TinyLinalg::numo_cDFloatId, double, double, TinyLinalg::DGESVD>::define_module_function(rb_mTinyLinalgLapack, "dgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cSFloatId, TinyLinalg::numo_cSFloatId, float, float, TinyLinalg::SGESVD>::define_module_function(rb_mTinyLinalgLapack, "sgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZGESVD>::define_module_function(rb_mTinyLinalgLapack, "zgesvd");
  TinyLinalg::GESVD<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CGESVD>::define_module_function(rb_mTinyLinalgLapack, "cgesvd");
  TinyLinalg::GESDD<TinyLinalg::numo_cDFloatId, TinyLinalg::numo_cDFloatId, double, double, TinyLinalg::DGESDD>::define_module_function(rb_mTinyLinalgLapack, "dgesdd");
  TinyLinalg::GESDD<TinyLinalg::numo_cSFloatId, TinyLinalg::numo_cSFloatId, float, float, TinyLinalg::SGESDD>::define_module_function(rb_mTinyLinalgLapack, "sgesdd");
  TinyLinalg::GESDD<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZGESDD>::define_module_function(rb_mTinyLinalgLapack, "zgesdd");
  TinyLinalg::GESDD<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CGESDD>::define_module_function(rb_mTinyLinalgLapack, "cgesdd");
}
