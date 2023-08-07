/**
 * Copyright (c) 2023 Atsushi Tatsuma
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "tiny_linalg.hpp"

#include "converter.hpp"
#include "util.hpp"

#include "blas/dot.hpp"
#include "blas/dot_sub.hpp"
#include "blas/gemm.hpp"
#include "blas/gemv.hpp"
#include "blas/nrm2.hpp"
#include "lapack/geqrf.hpp"
#include "lapack/gesdd.hpp"
#include "lapack/gesv.hpp"
#include "lapack/gesvd.hpp"
#include "lapack/getrf.hpp"
#include "lapack/getri.hpp"
#include "lapack/heev.hpp"
#include "lapack/heevd.hpp"
#include "lapack/hegv.hpp"
#include "lapack/hegvd.hpp"
#include "lapack/hegvx.hpp"
#include "lapack/orgqr.hpp"
#include "lapack/syev.hpp"
#include "lapack/syevd.hpp"
#include "lapack/sygv.hpp"
#include "lapack/sygvd.hpp"
#include "lapack/sygvx.hpp"
#include "lapack/ungqr.hpp"

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
   * @!visibility private
   */
  rb_mTinyLinalgBlas = rb_define_module_under(rb_mTinyLinalg, "Blas");
  /**
   * Document-module: Numo::TinyLinalg::Lapack
   * Numo::TinyLinalg::Lapack is wrapper module of LAPACK functions.
   * @!visibility private
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
  TinyLinalg::GeSv<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DGeSv>::define_module_function(rb_mTinyLinalgLapack, "dgesv");
  TinyLinalg::GeSv<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SGeSv>::define_module_function(rb_mTinyLinalgLapack, "sgesv");
  TinyLinalg::GeSv<TinyLinalg::numo_cDComplexId, lapack_complex_double, TinyLinalg::ZGeSv>::define_module_function(rb_mTinyLinalgLapack, "zgesv");
  TinyLinalg::GeSv<TinyLinalg::numo_cSComplexId, lapack_complex_float, TinyLinalg::CGeSv>::define_module_function(rb_mTinyLinalgLapack, "cgesv");
  TinyLinalg::GeSvd<TinyLinalg::numo_cDFloatId, TinyLinalg::numo_cDFloatId, double, double, TinyLinalg::DGeSvd>::define_module_function(rb_mTinyLinalgLapack, "dgesvd");
  TinyLinalg::GeSvd<TinyLinalg::numo_cSFloatId, TinyLinalg::numo_cSFloatId, float, float, TinyLinalg::SGeSvd>::define_module_function(rb_mTinyLinalgLapack, "sgesvd");
  TinyLinalg::GeSvd<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZGeSvd>::define_module_function(rb_mTinyLinalgLapack, "zgesvd");
  TinyLinalg::GeSvd<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CGeSvd>::define_module_function(rb_mTinyLinalgLapack, "cgesvd");
  TinyLinalg::GeSdd<TinyLinalg::numo_cDFloatId, TinyLinalg::numo_cDFloatId, double, double, TinyLinalg::DGeSdd>::define_module_function(rb_mTinyLinalgLapack, "dgesdd");
  TinyLinalg::GeSdd<TinyLinalg::numo_cSFloatId, TinyLinalg::numo_cSFloatId, float, float, TinyLinalg::SGeSdd>::define_module_function(rb_mTinyLinalgLapack, "sgesdd");
  TinyLinalg::GeSdd<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZGeSdd>::define_module_function(rb_mTinyLinalgLapack, "zgesdd");
  TinyLinalg::GeSdd<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CGeSdd>::define_module_function(rb_mTinyLinalgLapack, "cgesdd");
  TinyLinalg::GeTrf<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DGeTrf>::define_module_function(rb_mTinyLinalgLapack, "dgetrf");
  TinyLinalg::GeTrf<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SGeTrf>::define_module_function(rb_mTinyLinalgLapack, "sgetrf");
  TinyLinalg::GeTrf<TinyLinalg::numo_cDComplexId, lapack_complex_double, TinyLinalg::ZGeTrf>::define_module_function(rb_mTinyLinalgLapack, "zgetrf");
  TinyLinalg::GeTrf<TinyLinalg::numo_cSComplexId, lapack_complex_float, TinyLinalg::CGeTrf>::define_module_function(rb_mTinyLinalgLapack, "cgetrf");
  TinyLinalg::GeTri<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DGeTri>::define_module_function(rb_mTinyLinalgLapack, "dgetri");
  TinyLinalg::GeTri<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SGeTri>::define_module_function(rb_mTinyLinalgLapack, "sgetri");
  TinyLinalg::GeTri<TinyLinalg::numo_cDComplexId, lapack_complex_double, TinyLinalg::ZGeTri>::define_module_function(rb_mTinyLinalgLapack, "zgetri");
  TinyLinalg::GeTri<TinyLinalg::numo_cSComplexId, lapack_complex_float, TinyLinalg::CGeTri>::define_module_function(rb_mTinyLinalgLapack, "cgetri");
  TinyLinalg::GeQrf<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DGeQrf>::define_module_function(rb_mTinyLinalgLapack, "dgeqrf");
  TinyLinalg::GeQrf<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SGeQrf>::define_module_function(rb_mTinyLinalgLapack, "sgeqrf");
  TinyLinalg::GeQrf<TinyLinalg::numo_cDComplexId, lapack_complex_double, TinyLinalg::ZGeQrf>::define_module_function(rb_mTinyLinalgLapack, "zgeqrf");
  TinyLinalg::GeQrf<TinyLinalg::numo_cSComplexId, lapack_complex_float, TinyLinalg::CGeQrf>::define_module_function(rb_mTinyLinalgLapack, "cgeqrf");
  TinyLinalg::OrgQr<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DOrgQr>::define_module_function(rb_mTinyLinalgLapack, "dorgqr");
  TinyLinalg::OrgQr<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SOrgQr>::define_module_function(rb_mTinyLinalgLapack, "sorgqr");
  TinyLinalg::UngQr<TinyLinalg::numo_cDComplexId, lapack_complex_double, TinyLinalg::ZUngQr>::define_module_function(rb_mTinyLinalgLapack, "zungqr");
  TinyLinalg::UngQr<TinyLinalg::numo_cSComplexId, lapack_complex_float, TinyLinalg::CUngQr>::define_module_function(rb_mTinyLinalgLapack, "cungqr");
  TinyLinalg::SyEv<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DSyEv>::define_module_function(rb_mTinyLinalgLapack, "dsyev");
  TinyLinalg::SyEv<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SSyEv>::define_module_function(rb_mTinyLinalgLapack, "ssyev");
  TinyLinalg::HeEv<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZHeEv>::define_module_function(rb_mTinyLinalgLapack, "zheev");
  TinyLinalg::HeEv<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CHeEv>::define_module_function(rb_mTinyLinalgLapack, "cheev");
  TinyLinalg::SyEvd<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DSyEvd>::define_module_function(rb_mTinyLinalgLapack, "dsyevd");
  TinyLinalg::SyEvd<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SSyEvd>::define_module_function(rb_mTinyLinalgLapack, "ssyevd");
  TinyLinalg::HeEvd<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZHeEvd>::define_module_function(rb_mTinyLinalgLapack, "zheevd");
  TinyLinalg::HeEvd<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CHeEvd>::define_module_function(rb_mTinyLinalgLapack, "cheevd");
  TinyLinalg::SyGv<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DSyGv>::define_module_function(rb_mTinyLinalgLapack, "dsygv");
  TinyLinalg::SyGv<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SSyGv>::define_module_function(rb_mTinyLinalgLapack, "ssygv");
  TinyLinalg::HeGv<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZHeGv>::define_module_function(rb_mTinyLinalgLapack, "zhegv");
  TinyLinalg::HeGv<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CHeGv>::define_module_function(rb_mTinyLinalgLapack, "chegv");
  TinyLinalg::SyGvd<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DSyGvd>::define_module_function(rb_mTinyLinalgLapack, "dsygvd");
  TinyLinalg::SyGvd<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SSyGvd>::define_module_function(rb_mTinyLinalgLapack, "ssygvd");
  TinyLinalg::HeGvd<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZHeGvd>::define_module_function(rb_mTinyLinalgLapack, "zhegvd");
  TinyLinalg::HeGvd<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CHeGvd>::define_module_function(rb_mTinyLinalgLapack, "chegvd");
  TinyLinalg::SyGvx<TinyLinalg::numo_cDFloatId, double, TinyLinalg::DSyGvx>::define_module_function(rb_mTinyLinalgLapack, "dsygvx");
  TinyLinalg::SyGvx<TinyLinalg::numo_cSFloatId, float, TinyLinalg::SSyGvx>::define_module_function(rb_mTinyLinalgLapack, "ssygvx");
  TinyLinalg::HeGvx<TinyLinalg::numo_cDComplexId, TinyLinalg::numo_cDFloatId, lapack_complex_double, double, TinyLinalg::ZHeGvx>::define_module_function(rb_mTinyLinalgLapack, "zhegvx");
  TinyLinalg::HeGvx<TinyLinalg::numo_cSComplexId, TinyLinalg::numo_cSFloatId, lapack_complex_float, float, TinyLinalg::CHeGvx>::define_module_function(rb_mTinyLinalgLapack, "chegvx");

  rb_define_alias(rb_singleton_class(rb_mTinyLinalgBlas), "znrm2", "dznrm2");
  rb_define_alias(rb_singleton_class(rb_mTinyLinalgBlas), "cnrm2", "scnrm2");
}
