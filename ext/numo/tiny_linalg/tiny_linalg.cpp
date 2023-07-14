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

static VALUE tiny_linalg_blas_call(int argc, VALUE* argv, VALUE self) {
  VALUE fn_name = Qnil;
  VALUE nary_arr = Qnil;
  VALUE kw_args = Qnil;
  rb_scan_args(argc, argv, "1*:", &fn_name, &nary_arr, &kw_args);

  const char type = blas_char(nary_arr);
  if (type == 'n') {
    rb_raise(rb_eTypeError, "invalid data type for BLAS/LAPACK");
    return Qnil;
  }

  std::string type_str{ type };
  std::string fn_str = type_str + std::string(rb_id2name(rb_to_id(rb_to_symbol(fn_name))));
  ID fn_id = rb_intern(fn_str.c_str());
  size_t n = RARRAY_LEN(nary_arr);
  VALUE ret = Qnil;

  if (NIL_P(kw_args)) {
    VALUE* args = ALLOCA_N(VALUE, n);
    for (size_t i = 0; i < n; i++) {
      args[i] = rb_ary_entry(nary_arr, i);
    }
    ret = rb_funcallv(self, fn_id, n, args);
  } else {
    VALUE* args = ALLOCA_N(VALUE, n + 1);
    for (size_t i = 0; i < n; i++) {
      args[i] = rb_ary_entry(nary_arr, i);
    }
    args[n] = kw_args;
    ret = rb_funcallv_kw(self, fn_id, n + 1, args, RB_PASS_KEYWORDS);
  }

  return ret;
}

static VALUE tiny_linalg_dot(VALUE self, VALUE a_, VALUE b_) {
  VALUE a = IsNArray(a_) ? a_ : rb_funcall(numo_cNArray, rb_intern("asarray"), 1, a_);
  VALUE b = IsNArray(b_) ? b_ : rb_funcall(numo_cNArray, rb_intern("asarray"), 1, b_);

  VALUE arg_arr = rb_ary_new3(2, a, b);
  const char type = blas_char(arg_arr);
  if (type == 'n') {
    rb_raise(rb_eTypeError, "invalid data type for BLAS/LAPACK");
    return Qnil;
  }

  VALUE ret = Qnil;
  narray_t* a_nary = NULL;
  narray_t* b_nary = NULL;
  GetNArray(a, a_nary);
  GetNArray(b, b_nary);
  const int a_ndim = NA_NDIM(a_nary);
  const int b_ndim = NA_NDIM(b_nary);

  if (a_ndim == 1) {
    if (b_ndim == 1) {
      ID fn_id = type == 'c' || type == 'z' ? rb_intern("dotu") : rb_intern("dot");
      ret = rb_funcall(rb_mTinyLinalgBlas, rb_intern("call"), 3, ID2SYM(fn_id), a, b);
    } else {
      VALUE kw_args = rb_hash_new();
      if (!RTEST(nary_check_contiguous(b)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        b = rb_funcall(b, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("N"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("T"));
      }
      char fn_name[] = "xgemv";
      fn_name[0] = type;
      VALUE argv[3] = { b, a, kw_args };
      ret = rb_funcallv_kw(rb_mTinyLinalgBlas, rb_intern(fn_name), 3, argv, RB_PASS_KEYWORDS);
    }
  } else {
    if (b_ndim == 1) {
      VALUE kw_args = rb_hash_new();
      if (!RTEST(nary_check_contiguous(a)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        a = rb_funcall(a, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("T"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("trans")), rb_str_new_cstr("N"));
      }
      char fn_name[] = "xgemv";
      fn_name[0] = type;
      VALUE argv[3] = { a, b, kw_args };
      ret = rb_funcallv_kw(rb_mTinyLinalgBlas, rb_intern(fn_name), 3, argv, RB_PASS_KEYWORDS);
    } else {
      VALUE kw_args = rb_hash_new();
      if (!RTEST(nary_check_contiguous(a)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        a = rb_funcall(a, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transa")), rb_str_new_cstr("T"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transa")), rb_str_new_cstr("N"));
      }
      if (!RTEST(nary_check_contiguous(b)) && RTEST(rb_funcall(b, rb_intern("fortran_contiguous?"), 0))) {
        b = rb_funcall(b, rb_intern("transpose"), 0);
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transb")), rb_str_new_cstr("T"));
      } else {
        rb_hash_aset(kw_args, ID2SYM(rb_intern("transb")), rb_str_new_cstr("N"));
      }
      char fn_name[] = "xgemm";
      fn_name[0] = type;
      VALUE argv[3] = { a, b, kw_args };
      ret = rb_funcallv_kw(rb_mTinyLinalgBlas, rb_intern(fn_name), 3, argv, RB_PASS_KEYWORDS);
    }
  }

  RB_GC_GUARD(a);
  RB_GC_GUARD(b);

  return ret;
}

extern "C" void Init_tiny_linalg(void) {
  rb_require("numo/narray");

  /**
   * Document-module: Numo::TinyLinalg
   * Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
   */
  rb_mTinyLinalg = rb_define_module_under(rb_mNumo, "TinyLinalg");
  /**
   * Document-module: Numo::TinyLinalg::Blas
   * Numo::TinyLinalg::Blas is wrapper module of BLAS functions.
   */
  rb_mTinyLinalgBlas = rb_define_module_under(rb_mTinyLinalg, "Blas");
  /**
   * Document-module: Numo::TinyLinalg::Lapack
   * Numo::TinyLinalg::Lapack is wrapper module of LAPACK functions.
   */
  rb_mTinyLinalgLapack = rb_define_module_under(rb_mTinyLinalg, "Lapack");

  /**
   * Returns BLAS char ([sdcz]) defined by data-type of arguments.
   *
   * @overload blas_char(a, ...) -> String
   *   @param [Numo::NArray] a
   *   @return [String]
   */
  rb_define_module_function(rb_mTinyLinalg, "blas_char", RUBY_METHOD_FUNC(tiny_linalg_blas_char), -1);
  /**
   * Calculates dot product of two vectors / matrices.
   *
   * @overload dot(a, b) -> [Float|Complex|Numo::NArray]
   *   @param [Numo::NArray] a
   *   @param [Numo::NArray] b
   *   @return [Float|Complex|Numo::NArray]
   */
  rb_define_module_function(rb_mTinyLinalg, "dot", RUBY_METHOD_FUNC(tiny_linalg_dot), 2);
  /**
   * Calls BLAS function prefixed with BLAS char.
   *
   * @overload call(func, *args)
   *   @param func [Symbol] BLAS function name without BLAS char.
   *   @param args arguments of BLAS function.
   * @example
   *   Numo::TinyLinalg::Blas.call(:gemv, a, b)
   */
  rb_define_singleton_method(rb_mTinyLinalgBlas, "call", RUBY_METHOD_FUNC(tiny_linalg_blas_call), -1);

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

  rb_define_alias(rb_singleton_class(rb_mTinyLinalgBlas), "znrm2", "dznrm2");
  rb_define_alias(rb_singleton_class(rb_mTinyLinalgBlas), "cnrm2", "scnrm2");
}
