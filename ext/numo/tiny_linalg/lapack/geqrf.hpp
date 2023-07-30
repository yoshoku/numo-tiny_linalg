namespace TinyLinalg {

struct DGeQrf {
  lapack_int call(int matrix_layout, lapack_int m, lapack_int n,
                  double* a, lapack_int lda, double* tau) {
    return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
  }
};

struct SGeQrf {
  lapack_int call(int matrix_layout, lapack_int m, lapack_int n,
                  float* a, lapack_int lda, float* tau) {
    return LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau);
  }
};

struct ZGeQrf {
  lapack_int call(int matrix_layout, lapack_int m, lapack_int n,
                  lapack_complex_double* a, lapack_int lda, lapack_complex_double* tau) {
    return LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);
  }
};

struct CGeQrf {
  lapack_int call(int matrix_layout, lapack_int m, lapack_int n,
                  lapack_complex_float* a, lapack_int lda, lapack_complex_float* tau) {
    return LAPACKE_cgeqrf(matrix_layout, m, n, a, lda, tau);
  }
};

template <int nary_dtype_id, typename DType, typename FncType>
class GeQrf {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_geqrf), -1);
  }

private:
  struct geqrf_opt {
    int matrix_layout;
  };

  static void iter_geqrf(na_loop_t* const lp) {
    DType* a = (DType*)NDL_PTR(lp, 0);
    DType* tau = (DType*)NDL_PTR(lp, 1);
    int* info = (int*)NDL_PTR(lp, 2);
    geqrf_opt* opt = (geqrf_opt*)(lp->opt_ptr);
    const lapack_int m = NDL_SHAPE(lp, 0)[0];
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int lda = n;
    const lapack_int i = FncType().call(opt->matrix_layout, m, n, a, lda, tau);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_geqrf(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);
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

    narray_t* a_nary = NULL;
    GetNArray(a_vnary, a_nary);
    const int n_dims = NA_NDIM(a_nary);
    if (n_dims != 2) {
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");
      return Qnil;
    }

    size_t m = NA_SHAPE(a_nary)[0];
    size_t n = NA_SHAPE(a_nary)[1];
    size_t shape[1] = { m < n ? m : n };
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[2] = { { nary_dtype, 1, shape }, { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_geqrf, NO_LOOP | NDF_EXTRACT, 1, 2, ain, aout };
    geqrf_opt opt = { matrix_layout };
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);

    VALUE ret = rb_ary_concat(rb_ary_new3(1, a_vnary), res);

    RB_GC_GUARD(a_vnary);

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
