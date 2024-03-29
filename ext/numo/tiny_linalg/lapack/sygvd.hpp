namespace TinyLinalg {

struct DSyGvd {
  lapack_int call(int matrix_layout, lapack_int itype, char jobz,
                  char uplo, lapack_int n, double* a, lapack_int lda,
                  double* b, lapack_int ldb, double* w) {
    return LAPACKE_dsygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
  }
};

struct SSyGvd {
  lapack_int call(int matrix_layout, lapack_int itype, char jobz,
                  char uplo, lapack_int n, float* a, lapack_int lda,
                  float* b, lapack_int ldb, float* w) {
    return LAPACKE_ssygvd(matrix_layout, itype, jobz, uplo, n, a, lda, b, ldb, w);
  }
};

template <int nary_dtype_id, typename dtype, class LapackFn>
class SyGvd {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_sygvd), -1);
  }

private:
  struct sygvd_opt {
    int matrix_layout;
    lapack_int itype;
    char jobz;
    char uplo;
  };

  static void iter_sygvd(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    dtype* b = (dtype*)NDL_PTR(lp, 1);
    dtype* w = (dtype*)NDL_PTR(lp, 2);
    int* info = (int*)NDL_PTR(lp, 3);
    sygvd_opt* opt = (sygvd_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int lda = NDL_SHAPE(lp, 0)[0];
    const lapack_int ldb = NDL_SHAPE(lp, 1)[0];
    const lapack_int i = LapackFn().call(opt->matrix_layout, opt->itype, opt->jobz, opt->uplo, n, a, lda, b, ldb, w);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_sygvd(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE b_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);
    ID kw_table[4] = { rb_intern("itype"), rb_intern("jobz"), rb_intern("uplo"), rb_intern("order") };
    VALUE kw_values[4] = { Qundef, Qundef, Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 4, kw_values);
    const lapack_int itype = kw_values[0] != Qundef ? Util().get_itype(kw_values[0]) : 1;
    const char jobz = kw_values[1] != Qundef ? Util().get_jobz(kw_values[1]) : 'V';
    const char uplo = kw_values[2] != Qundef ? Util().get_uplo(kw_values[2]) : 'U';
    const int matrix_layout = kw_values[3] != Qundef ? Util().get_matrix_layout(kw_values[3]) : LAPACK_ROW_MAJOR;

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

    narray_t* a_nary = nullptr;
    GetNArray(a_vnary, a_nary);
    if (NA_NDIM(a_nary) != 2) {
      rb_raise(rb_eArgError, "input array a must be 2-dimensional");
      return Qnil;
    }
    if (NA_SHAPE(a_nary)[0] != NA_SHAPE(a_nary)[1]) {
      rb_raise(rb_eArgError, "input array a must be square");
      return Qnil;
    }
    narray_t* b_nary = nullptr;
    GetNArray(b_vnary, b_nary);
    if (NA_NDIM(b_nary) != 2) {
      rb_raise(rb_eArgError, "input array b must be 2-dimensional");
      return Qnil;
    }
    if (NA_SHAPE(b_nary)[0] != NA_SHAPE(b_nary)[1]) {
      rb_raise(rb_eArgError, "input array b must be square");
      return Qnil;
    }

    const size_t n = NA_SHAPE(a_nary)[1];
    size_t shape[1] = { n };
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[2] = { { nary_dtype, 1, shape }, { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_sygvd, NO_LOOP | NDF_EXTRACT, 2, 2, ain, aout };
    sygvd_opt opt = { matrix_layout, itype, jobz, uplo };
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);
    VALUE ret = rb_ary_new3(4, a_vnary, b_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1));

    RB_GC_GUARD(a_vnary);
    RB_GC_GUARD(b_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
