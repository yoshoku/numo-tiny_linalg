namespace TinyLinalg {

struct DPoTrs {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                  const double* a, lapack_int lda, double* b, lapack_int ldb) {
    return LAPACKE_dpotrs(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
  }
};

struct SPoTrs {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                  const float* a, lapack_int lda, float* b, lapack_int ldb) {
    return LAPACKE_spotrs(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
  }
};

struct ZPoTrs {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                  const lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb) {
    return LAPACKE_zpotrs(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
  }
};

struct CPoTrs {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs,
                  const lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb) {
    return LAPACKE_cpotrs(matrix_layout, uplo, n, nrhs, a, lda, b, ldb);
  }
};

template <int nary_dtype_id, typename dtype, class LapackFn>
class PoTrs {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_potrs), -1);
  }

private:
  struct potrs_opt {
    int matrix_layout;
    char uplo;
  };

  static void iter_potrs(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    dtype* b = (dtype*)NDL_PTR(lp, 1);
    int* info = (int*)NDL_PTR(lp, 2);
    potrs_opt* opt = (potrs_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[0];
    const lapack_int nrhs = lp->args[1].ndim == 1 ? 1 : NDL_SHAPE(lp, 1)[1];
    const lapack_int lda = n;
    const lapack_int ldb = nrhs;
    const lapack_int i = LapackFn().call(opt->matrix_layout, opt->uplo, n, nrhs, a, lda, b, ldb);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_potrs(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE b_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);
    ID kw_table[2] = { rb_intern("order"), rb_intern("uplo") };
    VALUE kw_values[2] = { Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);
    const int matrix_layout = kw_values[0] != Qundef ? Util().get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;
    const char uplo = kw_values[1] != Qundef ? Util().get_uplo(kw_values[1]) : 'U';

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
    GetNArray(a_vnary, a_nary);
    if (NA_NDIM(a_nary) != 2) {
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");
      return Qnil;
    }
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {
      rb_raise(rb_eArgError, "input array a must be square");
      return Qnil;
    }
    narray_t* b_nary = NULL;
    GetNArray(b_vnary, b_nary);
    const int b_n_dims = NA_NDIM(b_nary);
    if (b_n_dims != 1 && b_n_dims != 2) {
      rb_raise(rb_eArgError, "input array b must be 1- or 2-dimensional");
      return Qnil;
    }

    lapack_int n = NA_SHAPE(a_nary)[0];
    lapack_int nb = NA_SHAPE(b_nary)[0];
    if (n != nb) {
      rb_raise(nary_eShapeError, "shape1[0](=%d) != shape2[0](=%d)", n, nb);
    }

    ndfunc_arg_in_t ain[2] = { { nary_dtype, 2 }, { OVERWRITE, b_n_dims } };
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_potrs, NO_LOOP | NDF_EXTRACT, 2, 1, ain, aout };
    potrs_opt opt = { matrix_layout, uplo };
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);
    VALUE ret = rb_ary_new3(2, b_vnary, res);

    RB_GC_GUARD(a_vnary);
    RB_GC_GUARD(b_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
