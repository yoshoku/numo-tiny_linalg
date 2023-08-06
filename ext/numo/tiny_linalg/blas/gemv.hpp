namespace TinyLinalg {

struct DGemv {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
            const double alpha, const double* a, const blasint lda,
            const double* x, const blasint incx, const double beta, double* y, const blasint incy) {
    cblas_dgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
};

struct SGemv {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
            const float alpha, const float* a, const blasint lda,
            const float* x, const blasint incx, const float beta, float* y, const blasint incy) {
    cblas_sgemv(order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
};

struct ZGemv {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
            const dcomplex alpha, const dcomplex* a, const blasint lda,
            const dcomplex* x, const blasint incx, const dcomplex beta, dcomplex* y, const blasint incy) {
    cblas_zgemv(order, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
  }
};

struct CGemv {
  void call(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans, const blasint m, const blasint n,
            const scomplex alpha, const scomplex* a, const blasint lda,
            const scomplex* x, const blasint incx, const scomplex beta, scomplex* y, const blasint incy) {
    cblas_cgemv(order, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy);
  }
};

template <int nary_dtype_id, typename dtype, class BlasFn, class Converter>
class Gemv {
public:
  static void define_module_function(VALUE mBlas, const char* modfn_name) {
    rb_define_module_function(mBlas, modfn_name, RUBY_METHOD_FUNC(tiny_linalg_gemv), -1);
  };

private:
  struct options {
    dtype alpha;
    dtype beta;
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE trans;
    blasint m;
    blasint n;
  };

  static void iter_gemv(na_loop_t* const lp) {
    const dtype* a = (dtype*)NDL_PTR(lp, 0);
    const dtype* x = (dtype*)NDL_PTR(lp, 1);
    dtype* y = (dtype*)NDL_PTR(lp, 2);
    const options* opt = (options*)(lp->opt_ptr);
    const blasint lda = opt->n;
    BlasFn().call(opt->order, opt->trans, opt->m, opt->n, opt->alpha, a, lda, x, 1, opt->beta, y, 1);
  };

  static VALUE tiny_linalg_gemv(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a = Qnil;
    VALUE x = Qnil;
    VALUE y = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "21:", &a, &x, &y, &kw_args);

    ID kw_table[4] = { rb_intern("alpha"), rb_intern("beta"), rb_intern("order"), rb_intern("trans") };
    VALUE kw_values[4] = { Qundef, Qundef, Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 4, kw_values);

    if (CLASS_OF(a) != nary_dtype) {
      a = rb_funcall(nary_dtype, rb_intern("cast"), 1, a);
    }
    if (!RTEST(nary_check_contiguous(a))) {
      a = nary_dup(a);
    }
    if (CLASS_OF(x) != nary_dtype) {
      x = rb_funcall(nary_dtype, rb_intern("cast"), 1, x);
    }
    if (!RTEST(nary_check_contiguous(x))) {
      x = nary_dup(x);
    }
    if (!NIL_P(y)) {
      if (CLASS_OF(y) != nary_dtype) {
        y = rb_funcall(nary_dtype, rb_intern("cast"), 1, y);
      }
      if (!RTEST(nary_check_contiguous(y))) {
        y = nary_dup(y);
      }
    }

    dtype alpha = kw_values[0] != Qundef ? Converter().to_dtype(kw_values[0]) : Converter().one();
    dtype beta = kw_values[1] != Qundef ? Converter().to_dtype(kw_values[1]) : Converter().zero();
    enum CBLAS_ORDER order = kw_values[2] != Qundef ? Util().get_cblas_order(kw_values[2]) : CblasRowMajor;
    enum CBLAS_TRANSPOSE trans = kw_values[3] != Qundef ? Util().get_cblas_trans(kw_values[3]) : CblasNoTrans;

    narray_t* a_nary = NULL;
    GetNArray(a, a_nary);
    narray_t* x_nary = NULL;
    GetNArray(x, x_nary);

    if (NA_NDIM(a_nary) != 2) {
      rb_raise(rb_eArgError, "a must be 2-dimensional");
      return Qnil;
    }
    if (NA_NDIM(x_nary) != 1) {
      rb_raise(rb_eArgError, "x must be 1-dimensional");
      return Qnil;
    }
    if (NA_SIZE(a_nary) == 0) {
      rb_raise(rb_eArgError, "a must not be empty");
      return Qnil;
    }
    if (NA_SIZE(x_nary) == 0) {
      rb_raise(rb_eArgError, "x must not be empty");
      return Qnil;
    }

    const blasint ma = NA_SHAPE(a_nary)[0];
    const blasint na = NA_SHAPE(a_nary)[1];
    const blasint mx = NA_SHAPE(x_nary)[0];
    const blasint m = trans == CblasNoTrans ? ma : na;
    const blasint n = trans == CblasNoTrans ? na : ma;

    if (n != mx) {
      rb_raise(nary_eShapeError, "shape1[1](=%d) != shape2[0](=%d)", n, mx);
      return Qnil;
    }

    options opt = { alpha, beta, order, trans, ma, na };
    size_t shape_out[1] = { static_cast<size_t>(m) };
    ndfunc_arg_out_t aout[1] = { { nary_dtype, 1, shape_out } };
    VALUE ret = Qnil;

    if (!NIL_P(y)) {
      narray_t* y_nary = NULL;
      GetNArray(y, y_nary);
      blasint my = NA_SHAPE(y_nary)[0];
      if (m > my) {
        rb_raise(nary_eShapeError, "shape3[0](=%d) >= shape1[0]=%d", my, m);
        return Qnil;
      }
      ndfunc_arg_in_t ain[3] = { { nary_dtype, 2 }, { nary_dtype, 1 }, { OVERWRITE, 1 } };
      ndfunc_t ndf = { iter_gemv, NO_LOOP, 3, 0, ain, aout };
      na_ndloop3(&ndf, &opt, 3, a, x, y);
      ret = y;
    } else {
      y = INT2NUM(0);
      ndfunc_arg_in_t ain[3] = { { nary_dtype, 2 }, { nary_dtype, 1 }, { sym_init, 0 } };
      ndfunc_t ndf = { iter_gemv, NO_LOOP, 3, 1, ain, aout };
      ret = na_ndloop3(&ndf, &opt, 3, a, x, y);
    }

    RB_GC_GUARD(a);
    RB_GC_GUARD(x);
    RB_GC_GUARD(y);

    return ret;
  }
};

} // namespace TinyLinalg
