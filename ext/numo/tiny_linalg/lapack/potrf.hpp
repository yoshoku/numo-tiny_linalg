namespace TinyLinalg {

struct DPoTrf {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda) {
    return LAPACKE_dpotrf(matrix_layout, uplo, n, a, lda);
  }
};

struct SPoTrf {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, float* a, lapack_int lda) {
    return LAPACKE_spotrf(matrix_layout, uplo, n, a, lda);
  }
};

struct ZPoTrf {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda) {
    return LAPACKE_zpotrf(matrix_layout, uplo, n, a, lda);
  }
};

struct CPoTrf {
  lapack_int call(int matrix_layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda) {
    return LAPACKE_cpotrf(matrix_layout, uplo, n, a, lda);
  }
};

template <int nary_dtype_id, typename dtype, class LapackFn>
class PoTrf {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_potrf), -1);
  }

private:
  struct potrf_opt {
    int matrix_layout;
    char uplo;
  };

  static void iter_potrf(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    int* info = (int*)NDL_PTR(lp, 1);
    potrf_opt* opt = (potrf_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[0];
    const lapack_int lda = NDL_SHAPE(lp, 0)[1];
    const lapack_int i = LapackFn().call(opt->matrix_layout, opt->uplo, n, a, lda);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_potrf(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);
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

    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[1] = { { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_potrf, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };
    potrf_opt opt = { matrix_layout, uplo };
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);
    VALUE ret = rb_ary_new3(2, a_vnary, res);

    RB_GC_GUARD(a_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
