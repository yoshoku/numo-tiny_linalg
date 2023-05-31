namespace TinyLinalg {

struct DDot {
  double call(const int n, const double* x, const int incx, const double* y, const int incy) {
    return cblas_ddot(n, x, incx, y, incy);
  }
};

struct SDot {
  float call(const int n, const float* x, const int incx, const float* y, const int incy) {
    return cblas_sdot(n, x, incx, y, incy);
  }
};

template <int nary_dtype_id, typename dtype, class BlasFn>
class Dot {
public:
  static void define_module_function(VALUE mBlas, const char* modfn_name) {
    rb_define_module_function(mBlas, modfn_name, RUBY_METHOD_FUNC(tiny_linalg_dot), 2);
  };

private:
  static void iter_dot(na_loop_t* const lp) {
    dtype* x = (dtype*)NDL_PTR(lp, 0);
    dtype* y = (dtype*)NDL_PTR(lp, 1);
    dtype* d = (dtype*)NDL_PTR(lp, 2);
    const size_t n = NDL_SHAPE(lp, 0)[0];
    dtype ret = BlasFn().call(n, x, 1, y, 1);
    *d = ret;
  };

  static VALUE tiny_linalg_dot(VALUE self, VALUE x, VALUE y) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    if (CLASS_OF(x) != nary_dtype) {
      x = rb_funcall(nary_dtype, rb_intern("cast"), 1, x);
    }
    if (!RTEST(nary_check_contiguous(x))) {
      x = nary_dup(x);
    }
    if (CLASS_OF(y) != nary_dtype) {
      y = rb_funcall(nary_dtype, rb_intern("cast"), 1, y);
    }
    if (!RTEST(nary_check_contiguous(y))) {
      y = nary_dup(y);
    }

    narray_t* x_nary = NULL;
    GetNArray(x, x_nary);
    narray_t* y_nary = NULL;
    GetNArray(y, y_nary);

    if (NA_NDIM(x_nary) != 1) {
      rb_raise(rb_eArgError, "x must be 1-dimensional");
      return Qnil;
    }
    if (NA_NDIM(y_nary) != 1) {
      rb_raise(rb_eArgError, "y must be 1-dimensional");
      return Qnil;
    }
    if (NA_SIZE(x_nary) == 0) {
      rb_raise(rb_eArgError, "x must not be empty");
      return Qnil;
    }
    if (NA_SIZE(y_nary) == 0) {
      rb_raise(rb_eArgError, "x must not be empty");
      return Qnil;
    }
    if (NA_SIZE(x_nary) != NA_SIZE(y_nary)) {
      rb_raise(rb_eArgError, "x and y must have same size");
      return Qnil;
    }

    ndfunc_arg_in_t ain[2] = { { nary_dtype, 1 }, { nary_dtype, 1 } };
    size_t shape_out[1] = { 1 };
    ndfunc_arg_out_t aout[1] = { { nary_dtype, 0, shape_out } };
    ndfunc_t ndf = { iter_dot, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };
    VALUE ret = na_ndloop(&ndf, 2, x, y);

    RB_GC_GUARD(x);
    RB_GC_GUARD(y);
    return ret;
  };
};

} // namespace TinyLinalg
