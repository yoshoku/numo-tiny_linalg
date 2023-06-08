namespace TinyLinalg {

struct DGemm {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
            const blasint m, const blasint n, const blasint k,
            const double alpha, const double* a, const blasint lda, const double* b, const blasint ldb, const double beta,
            double* c, const blasint ldc) {
    cblas_dgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
};

struct SGemm {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
            const blasint m, const blasint n, const blasint k,
            const float alpha, const float* a, const blasint lda, const float* b, const blasint ldb, const float beta,
            float* c, const blasint ldc) {
    cblas_sgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
};

struct ZGemm {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
            const blasint m, const blasint n, const blasint k,
            const dcomplex alpha, const dcomplex* a, const blasint lda, const dcomplex* b, const blasint ldb, const dcomplex beta,
            dcomplex* c, const blasint ldc) {
    cblas_zgemm(order, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  }
};

struct CGemm {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb,
            const blasint m, const blasint n, const blasint k,
            const scomplex alpha, const scomplex* a, const blasint lda, const scomplex* b, const blasint ldb, const scomplex beta,
            scomplex* c, const blasint ldc) {
    cblas_cgemm(order, transa, transb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
  }
};

template <int nary_dtype_id, typename dtype, class BlasFn, class Converter>
class Gemm {
public:
  static void define_module_function(VALUE mBlas, const char* modfn_name) {
    rb_define_module_function(mBlas, modfn_name, RUBY_METHOD_FUNC(tiny_linalg_gemm), -1);
  };

private:
  struct options {
    dtype alpha;
    dtype beta;
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE transa;
    enum CBLAS_TRANSPOSE transb;
    blasint m;
    blasint n;
    blasint k;
  };

  static void iter_gemm(na_loop_t* const lp) {
    const dtype* a = (dtype*)NDL_PTR(lp, 0);
    const dtype* b = (dtype*)NDL_PTR(lp, 1);
    dtype* c = (dtype*)NDL_PTR(lp, 2);
    const options* opt = (options*)(lp->opt_ptr);
    const blasint lda = opt->transa == CblasNoTrans ? opt->k : opt->m;
    const blasint ldb = opt->transb == CblasNoTrans ? opt->n : opt->k;
    const blasint ldc = opt->m;
    BlasFn().call(opt->order, opt->transa, opt->transb, opt->m, opt->n, opt->k, opt->alpha, a, lda, b, ldb, opt->beta, c, ldc);
  };

  static VALUE tiny_linalg_gemm(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a = Qnil;
    VALUE b = Qnil;
    VALUE c = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "21:", &a, &b, &c, &kw_args);

    ID kw_table[5] = { rb_intern("alpha"), rb_intern("beta"), rb_intern("order"), rb_intern("transa"), rb_intern("transb") };
    VALUE kw_values[5] = { Qundef, Qundef, Qundef, Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 5, kw_values);

    if (CLASS_OF(a) != nary_dtype) {
      a = rb_funcall(nary_dtype, rb_intern("cast"), 1, a);
    }
    if (!RTEST(nary_check_contiguous(a))) {
      a = nary_dup(a);
    }
    if (CLASS_OF(b) != nary_dtype) {
      b = rb_funcall(nary_dtype, rb_intern("cast"), 1, b);
    }
    if (!RTEST(nary_check_contiguous(b))) {
      b = nary_dup(b);
    }
    if (!NIL_P(c)) {
      if (CLASS_OF(c) != nary_dtype) {
        c = rb_funcall(nary_dtype, rb_intern("cast"), 1, c);
      }
      if (!RTEST(nary_check_contiguous(c))) {
        c = nary_dup(c);
      }
    }

    dtype alpha = kw_values[0] != Qundef ? Converter().to_dtype(kw_values[0]) : Converter().one();
    dtype beta = kw_values[1] != Qundef ? Converter().to_dtype(kw_values[1]) : Converter().zero();
    enum CBLAS_ORDER order = kw_values[2] != Qundef ? get_cblas_order(kw_values[2]) : CblasRowMajor;
    enum CBLAS_TRANSPOSE transa = kw_values[3] != Qundef ? get_cblas_trans(kw_values[3]) : CblasNoTrans;
    enum CBLAS_TRANSPOSE transb = kw_values[4] != Qundef ? get_cblas_trans(kw_values[4]) : CblasNoTrans;

    narray_t* a_nary = NULL;
    GetNArray(a, a_nary);
    narray_t* b_nary = NULL;
    GetNArray(b, b_nary);

    if (NA_NDIM(a_nary) != 2) {
      rb_raise(rb_eArgError, "a must be 2-dimensional");
      return Qnil;
    }
    if (NA_NDIM(b_nary) != 2) {
      rb_raise(rb_eArgError, "b must be 2-dimensional");
      return Qnil;
    }
    if (NA_SIZE(a_nary) == 0) {
      rb_raise(rb_eArgError, "a must not be empty");
      return Qnil;
    }
    if (NA_SIZE(b_nary) == 0) {
      rb_raise(rb_eArgError, "b must not be empty");
      return Qnil;
    }

    const blasint ma = NA_SHAPE(a_nary)[0];
    const blasint ka = NA_SHAPE(a_nary)[1];
    const blasint kb = NA_SHAPE(b_nary)[0];
    const blasint nb = NA_SHAPE(b_nary)[1];
    const blasint m = transa == CblasNoTrans ? ma : ka;
    const blasint n = transb == CblasNoTrans ? nb : kb;
    const blasint k = transa == CblasNoTrans ? ka : ma;
    const blasint l = transb == CblasNoTrans ? kb : nb;

    if (k != l) {
      rb_raise(nary_eShapeError, "shape1[1](=%d) != shape2[0](=%d)", k, l);
      return Qnil;
    }

    options opt = { alpha, beta, order, transa, transb, m, n, k };
    size_t shape_out[2] = { static_cast<size_t>(m), static_cast<size_t>(n) };
    ndfunc_arg_out_t aout[1] = { { nary_dtype, 2, shape_out } };
    VALUE ret = Qnil;

    if (!NIL_P(c)) {
      narray_t* c_nary = NULL;
      GetNArray(c, c_nary);
      blasint nc = NA_SHAPE(c_nary)[0];
      if (m > nc) {
        rb_raise(nary_eShapeError, "shape3[0](=%d) >= shape1[0]=%d", nc, m);
        return Qnil;
      }
      ndfunc_arg_in_t ain[3] = { { nary_dtype, 2 }, { nary_dtype, 2 }, { OVERWRITE, 2 } };
      ndfunc_t ndf = { iter_gemm, NO_LOOP, 3, 0, ain, aout };
      na_ndloop3(&ndf, &opt, 3, a, b, c);
      ret = c;
    } else {
      c = INT2NUM(0);
      ndfunc_arg_in_t ain[3] = { { nary_dtype, 2 }, { nary_dtype, 2 }, { sym_init, 0 } };
      ndfunc_t ndf = { iter_gemm, NO_LOOP, 3, 1, ain, aout };
      ret = na_ndloop3(&ndf, &opt, 3, a, b, c);
    }

    RB_GC_GUARD(a);
    RB_GC_GUARD(b);
    RB_GC_GUARD(c);

    return ret;
  };

  static enum CBLAS_TRANSPOSE get_cblas_trans(VALUE val) {
    const char* option_str = StringValueCStr(val);
    enum CBLAS_TRANSPOSE res = CblasNoTrans;

    if (std::strlen(option_str) > 0) {
      switch (option_str[0]) {
      case 'n':
      case 'N':
        res = CblasNoTrans;
        break;
      case 't':
      case 'T':
        res = CblasTrans;
        break;
      case 'c':
      case 'C':
        res = CblasConjTrans;
        break;
      }
    }

    RB_GC_GUARD(val);

    return res;
  }

  static enum CBLAS_ORDER get_cblas_order(VALUE val) {
    const char* option_str = StringValueCStr(val);

    if (std::strlen(option_str) > 0) {
      switch (option_str[0]) {
      case 'r':
      case 'R':
        break;
      case 'c':
      case 'C':
        rb_warn("Numo::TinyLinalg::BLAS.gemm does not support column major.");
        break;
      }
    }

    RB_GC_GUARD(val);

    return CblasRowMajor;
  }
};

} // namespace TinyLinalg
