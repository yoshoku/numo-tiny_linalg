namespace TinyLinalg {

struct ZHeEvd {
  lapack_int call(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, double* w) {
    return LAPACKE_zheevd(matrix_layout, jobz, uplo, n, a, lda, w);
  }
};

struct CHeEvd {
  lapack_int call(int matrix_layout, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, float* w) {
    return LAPACKE_cheevd(matrix_layout, jobz, uplo, n, a, lda, w);
  }
};

template <int nary_dtype_id, int nary_rtype_id, typename dtype, typename rtype, class LapackFn>
class HeEvd {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_heevd), -1);
  }

private:
  struct heevd_opt {
    int matrix_layout;
    char jobz;
    char uplo;
  };

  static void iter_heevd(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    rtype* w = (rtype*)NDL_PTR(lp, 1);
    int* info = (int*)NDL_PTR(lp, 2);
    heevd_opt* opt = (heevd_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int lda = NDL_SHAPE(lp, 0)[0];
    const lapack_int i = LapackFn().call(opt->matrix_layout, opt->jobz, opt->uplo, n, a, lda, w);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_heevd(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];
    VALUE nary_rtype = NaryTypes[nary_rtype_id];

    VALUE a_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);
    ID kw_table[3] = { rb_intern("jobz"), rb_intern("uplo"), rb_intern("order") };
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);
    const char jobz = kw_values[0] != Qundef ? Util().get_jobz(kw_values[0]) : 'V';
    const char uplo = kw_values[1] != Qundef ? Util().get_uplo(kw_values[1]) : 'U';
    const int matrix_layout = kw_values[2] != Qundef ? Util().get_matrix_layout(kw_values[2]) : LAPACK_ROW_MAJOR;

    if (CLASS_OF(a_vnary) != nary_dtype) {
      a_vnary = rb_funcall(nary_dtype, rb_intern("cast"), 1, a_vnary);
    }
    if (!RTEST(nary_check_contiguous(a_vnary))) {
      a_vnary = nary_dup(a_vnary);
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

    const size_t n = NA_SHAPE(a_nary)[1];
    size_t shape[1] = { n };
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[2] = { { nary_rtype, 1, shape }, { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_heevd, NO_LOOP | NDF_EXTRACT, 1, 2, ain, aout };
    heevd_opt opt = { matrix_layout, jobz, uplo };
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);
    VALUE ret = rb_ary_new3(3, a_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1));

    RB_GC_GUARD(a_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
