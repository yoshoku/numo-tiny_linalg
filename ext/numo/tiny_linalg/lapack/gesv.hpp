namespace TinyLinalg {

struct DGeSv {
  lapack_int call(int matrix_layout, lapack_int n, lapack_int nrhs,
                  double* a, lapack_int lda, lapack_int* ipiv,
                  double* b, lapack_int ldb) {
    return LAPACKE_dgesv(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb);
  }
};

struct SGeSv {
  lapack_int call(int matrix_layout, lapack_int n, lapack_int nrhs,
                  float* a, lapack_int lda, lapack_int* ipiv,
                  float* b, lapack_int ldb) {
    return LAPACKE_sgesv(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb);
  }
};

struct ZGeSv {
  lapack_int call(int matrix_layout, lapack_int n, lapack_int nrhs,
                  lapack_complex_double* a, lapack_int lda, lapack_int* ipiv,
                  lapack_complex_double* b, lapack_int ldb) {
    return LAPACKE_zgesv(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb);
  }
};

struct CGeSv {
  lapack_int call(int matrix_layout, lapack_int n, lapack_int nrhs,
                  lapack_complex_float* a, lapack_int lda, lapack_int* ipiv,
                  lapack_complex_float* b, lapack_int ldb) {
    return LAPACKE_cgesv(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb);
  }
};

template <int nary_dtype_id, typename dtype, class LapackFn>
class GeSv {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_gesv), -1);
  }

private:
  struct gesv_opt {
    int matrix_layout;
  };

  static void iter_gesv(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    dtype* b = (dtype*)NDL_PTR(lp, 1);
    int* ipiv = (int*)NDL_PTR(lp, 2);
    int* info = (int*)NDL_PTR(lp, 3);
    gesv_opt* opt = (gesv_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[0];
    const lapack_int nhrs = lp->args[1].ndim == 1 ? 1 : NDL_SHAPE(lp, 1)[1];
    const lapack_int lda = n;
    const lapack_int ldb = nhrs;
    const lapack_int i = LapackFn().call(opt->matrix_layout, n, nhrs, a, lda, ipiv, b, ldb);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_gesv(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];
    VALUE a_vnary = Qnil;
    VALUE b_vnary = Qnil;
    VALUE kw_args = Qnil;

    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);

    ID kw_table[1] = { rb_intern("order") };
    VALUE kw_values[1] = { Qundef };

    rb_get_kwargs(kw_args, kw_table, 0, 1, kw_values);

    const int matrix_layout = kw_values[0] != Qundef ? Util().get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;

    if (CLASS_OF(a_vnary) != nary_dtype) {
      a_vnary = rb_funcall(nary_dtype, rb_intern("cast"), 1, a_vnary);
    }
    if (!RTEST(nary_check_contiguous(a_vnary))) {
      a_vnary = nary_dup(a_vnary);
    }
    if (CLASS_OF(b_vnary) != nary_dtype) {
      b_vnary = rb_funcall(nary_dtype, rb_intern("cast"), 1, b_vnary);
    }
    if (!RTEST(nary_check_contiguous(b_vnary))) {
      b_vnary = nary_dup(b_vnary);
    }

    narray_t* a_nary = NULL;
    narray_t* b_nary = NULL;
    GetNArray(a_vnary, a_nary);
    GetNArray(b_vnary, b_nary);
    const int a_n_dims = NA_NDIM(a_nary);
    const int b_n_dims = NA_NDIM(b_nary);
    if (a_n_dims != 2) {
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");
      return Qnil;
    }
    if (b_n_dims != 1 && b_n_dims != 2) {
      rb_raise(rb_eArgError, "input array b must be 1- or 2-dimensional");
      return Qnil;
    }

    lapack_int n = NA_SHAPE(a_nary)[0];
    lapack_int nb = b_n_dims == 1 ? NA_SHAPE(b_nary)[0] : NA_SHAPE(b_nary)[0];
    if (n != nb) {
      rb_raise(nary_eShapeError, "shape1[1](=%d) != shape2[0](=%d)", n, nb);
    }

    lapack_int nhrs = b_n_dims == 1 ? 1 : NA_SHAPE(b_nary)[1];
    size_t shape[2] = { static_cast<size_t>(n), static_cast<size_t>(nhrs) };
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, b_n_dims } };
    ndfunc_arg_out_t aout[2] = { { numo_cInt32, 1, shape }, { numo_cInt32, 0 } };

    ndfunc_t ndf = { iter_gesv, NO_LOOP | NDF_EXTRACT, 2, 2, ain, aout };
    gesv_opt opt = { matrix_layout };
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);

    VALUE ret = rb_ary_concat(rb_assoc_new(a_vnary, b_vnary), res);

    RB_GC_GUARD(a_vnary);
    RB_GC_GUARD(b_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
