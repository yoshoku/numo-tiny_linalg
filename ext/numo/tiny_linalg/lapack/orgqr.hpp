namespace TinyLinalg {

struct DOrgQr {
  lapack_int call(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                  double* a, lapack_int lda, const double* tau) {
    return LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau);
  }
};

struct SOrgQr {
  lapack_int call(int matrix_layout, lapack_int m, lapack_int n, lapack_int k,
                  float* a, lapack_int lda, const float* tau) {
    return LAPACKE_sorgqr(matrix_layout, m, n, k, a, lda, tau);
  }
};

template <int nary_dtype_id, typename DType, typename FncType>
class OrgQr {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_orgqr), -1);
  }

private:
  struct orgqr_opt {
    int matrix_layout;
  };

  static void iter_orgqr(na_loop_t* const lp) {
    DType* a = (DType*)NDL_PTR(lp, 0);
    DType* tau = (DType*)NDL_PTR(lp, 1);
    int* info = (int*)NDL_PTR(lp, 2);
    orgqr_opt* opt = (orgqr_opt*)(lp->opt_ptr);
    const lapack_int m = NDL_SHAPE(lp, 0)[0];
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int k = NDL_SHAPE(lp, 1)[0];
    const lapack_int lda = n;
    const lapack_int i = FncType().call(opt->matrix_layout, m, n, k, a, lda, tau);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_orgqr(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE tau_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "2:", &a_vnary, &tau_vnary, &kw_args);
    ID kw_table[1] = { rb_intern("order") };
    VALUE kw_values[1] = { Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 1, kw_values);
    const int matrix_layout = kw_values[0] != Qundef ? get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;

    if (CLASS_OF(a_vnary) != nary_dtype) {
      a_vnary = rb_funcall(nary_dtype, rb_intern("cast"), 1, a_vnary);
    }
    if (!RTEST(nary_check_contiguous(a_vnary))) {
      a_vnary = nary_dup(a_vnary);
    }
    if (CLASS_OF(tau_vnary) != nary_dtype) {
      tau_vnary = rb_funcall(nary_dtype, rb_intern("cast"), 1, tau_vnary);
    }
    if (!RTEST(nary_check_contiguous(tau_vnary))) {
      tau_vnary = nary_dup(tau_vnary);
    }

    narray_t* a_nary = NULL;
    GetNArray(a_vnary, a_nary);
    if (NA_NDIM(a_nary) != 2) {
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");
      return Qnil;
    }
    narray_t* tau_nary = NULL;
    GetNArray(tau_vnary, tau_nary);
    if (NA_NDIM(tau_nary) != 1) {
      rb_raise(rb_eArgError, "input array tau must be 1-dimensional");
      return Qnil;
    }

    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { nary_dtype, 1 } };
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_orgqr, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };
    orgqr_opt opt = { matrix_layout };
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, tau_vnary);

    VALUE ret = rb_ary_new3(2, a_vnary, res);

    RB_GC_GUARD(a_vnary);
    RB_GC_GUARD(tau_vnary);

    return ret;
  }

  static int get_matrix_layout(VALUE val) {
    const char* option_str = StringValueCStr(val);

    if (std::strlen(option_str) > 0) {
      switch (option_str[0]) {
      case 'r':
      case 'R':
        break;
      case 'c':
      case 'C':
        rb_warn("Numo::TinyLinalg::Lapack.getrf does not support column major.");
        break;
      }
    }

    RB_GC_GUARD(val);

    return LAPACK_ROW_MAJOR;
  }
};

} // namespace TinyLinalg
