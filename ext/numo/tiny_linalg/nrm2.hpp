namespace TinyLinalg {

struct DNrm2 {
  double call(const int n, const double* x, const int incx) {
    return cblas_dnrm2(n, x, incx);
  }
};

struct SNrm2 {
  float call(const int n, const float* x, const int incx) {
    return cblas_snrm2(n, x, incx);
  }
};

template <int nary_dtype_id, typename dtype, class BlasFn>
class Nrm2 {
public:
  static void define_module_function(VALUE mBlas, const char* modfn_name) {
    rb_define_module_function(mBlas, modfn_name, RUBY_METHOD_FUNC(tiny_linalg_dot), -1);
  }

private:
  static void iter_nrm2(na_loop_t* const lp) {
    dtype* x = (dtype*)NDL_PTR(lp, 0);
    dtype* d = (dtype*)NDL_PTR(lp, 1);
    const size_t n = NDL_SHAPE(lp, 0)[0];
    dtype ret = BlasFn().call(n, x, 1);
    *d = ret;
  }

  static VALUE tiny_linalg_dot(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE x = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "1:", &x, &kw_args);

    ID kw_table[1] = { rb_intern("keepdims") };
    VALUE kw_values[1] = { Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 1, kw_values);
    const bool keepdims = kw_values[0] != Qundef ? RTEST(kw_values[0]) : false;

    if (CLASS_OF(x) != nary_dtype) {
      x = rb_funcall(nary_dtype, rb_intern("cast"), 1, x);
    }
    if (!RTEST(nary_check_contiguous(x))) {
      x = nary_dup(x);
    }

    narray_t* x_nary = NULL;
    GetNArray(x, x_nary);

    if (NA_NDIM(x_nary) != 1) {
      rb_raise(rb_eArgError, "x must be 1-dimensional");
      return Qnil;
    }
    if (NA_SIZE(x_nary) == 0) {
      rb_raise(rb_eArgError, "x must not be empty");
      return Qnil;
    }

    ndfunc_arg_in_t ain[1] = { { nary_dtype, 1 } };
    size_t shape_out[1] = { 1 };
    ndfunc_arg_out_t aout[1] = { { nary_dtype, 0, shape_out } };
    ndfunc_t ndf = { iter_nrm2, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };
    if (keepdims) {
      ndf.flag |= NDF_KEEP_DIM;
    }

    VALUE ret = na_ndloop(&ndf, 1, x);

    RB_GC_GUARD(x);
    return ret;
  }
};

} // namespace TinyLinalg
