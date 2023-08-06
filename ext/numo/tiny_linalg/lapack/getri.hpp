namespace TinyLinalg {

struct DGeTri {
  lapack_int call(int matrix_layout, lapack_int n, double* a, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_dgetri(matrix_layout, n, a, lda, ipiv);
  }
};

struct SGeTri {
  lapack_int call(int matrix_layout, lapack_int n, float* a, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_sgetri(matrix_layout, n, a, lda, ipiv);
  }
};

struct ZGeTri {
  lapack_int call(int matrix_layout, lapack_int n, lapack_complex_double* a, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_zgetri(matrix_layout, n, a, lda, ipiv);
  }
};

struct CGeTri {
  lapack_int call(int matrix_layout, lapack_int n, lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv) {
    return LAPACKE_cgetri(matrix_layout, n, a, lda, ipiv);
  }
};

template <int nary_dtype_id, typename dtype, class LapackFn>
class GeTri {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_getri), -1);
  }

private:
  struct getri_opt {
    int matrix_layout;
  };

  static void iter_getri(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    lapack_int* ipiv = (lapack_int*)NDL_PTR(lp, 1);
    int* info = (int*)NDL_PTR(lp, 2);
    getri_opt* opt = (getri_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[0];
    const lapack_int lda = n;
    const lapack_int i = LapackFn().call(opt->matrix_layout, n, a, lda, ipiv);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_getri(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE ipiv_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "2:", &a_vnary, &ipiv_vnary, &kw_args);
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
    if (CLASS_OF(ipiv_vnary) != numo_cInt32) {
      ipiv_vnary = rb_funcall(numo_cInt32, rb_intern("cast"), 1, ipiv_vnary);
    }
    if (!RTEST(nary_check_contiguous(ipiv_vnary))) {
      ipiv_vnary = nary_dup(ipiv_vnary);
    }

    narray_t* a_nary = NULL;
    GetNArray(a_vnary, a_nary);
    if (NA_NDIM(a_nary) != 2) {
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");
      return Qnil;
    }
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {
      rb_raise(rb_eArgError, "input array a must be square");
      return Qnil;
    }
    narray_t* ipiv_nary = NULL;
    GetNArray(ipiv_vnary, ipiv_nary);
    if (NA_NDIM(ipiv_nary) != 1) {
      rb_raise(rb_eArgError, "input array ipiv must be 1-dimensional");
      return Qnil;
    }

    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { numo_cInt32, 1 } };
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_getri, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };
    getri_opt opt = { matrix_layout };
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, ipiv_vnary);

    VALUE ret = rb_ary_new3(2, a_vnary, res);

    RB_GC_GUARD(a_vnary);
    RB_GC_GUARD(ipiv_vnary);

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
        rb_warn("Numo::TinyLinalg::Lapack.getri does not support column major.");
        break;
      }
    }

    RB_GC_GUARD(val);

    return LAPACK_ROW_MAJOR;
  }
};

} // namespace TinyLinalg
