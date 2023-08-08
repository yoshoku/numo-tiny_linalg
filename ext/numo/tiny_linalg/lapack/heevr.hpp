namespace TinyLinalg {

struct ZHeEvr {
  lapack_int call(int matrix_layout, char jobz, char range, char uplo,
                  lapack_int n, lapack_complex_double* a, lapack_int lda, double vl, double vu, lapack_int il,
                  lapack_int iu, double abstol, lapack_int* m,
                  double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* isuppz) {
    return LAPACKE_zheevr(matrix_layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz);
  }
};

struct CHeEvr {
  lapack_int call(int matrix_layout, char jobz, char range, char uplo,
                  lapack_int n, lapack_complex_float* a, lapack_int lda, float vl, float vu, lapack_int il,
                  lapack_int iu, float abstol, lapack_int* m,
                  float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* isuppz) {
    return LAPACKE_cheevr(matrix_layout, jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, z, ldz, isuppz);
  }
};

template <int nary_dtype_id, int nary_rtype_id, typename dtype, typename rtype, class LapackFn>
class HeEvr {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_heevr), -1);
  }

private:
  struct heevr_opt {
    int matrix_layout;
    char jobz;
    char range;
    char uplo;
    rtype vl;
    rtype vu;
    lapack_int il;
    lapack_int iu;
  };

  static void iter_heevr(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    int* m = (int*)NDL_PTR(lp, 1);
    rtype* w = (rtype*)NDL_PTR(lp, 2);
    dtype* z = (dtype*)NDL_PTR(lp, 3);
    int* isuppz = (int*)NDL_PTR(lp, 4);
    int* info = (int*)NDL_PTR(lp, 5);
    heevr_opt* opt = (heevr_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int lda = NDL_SHAPE(lp, 0)[0];
    const lapack_int ldz = opt->range != 'I' ? n : opt->iu - opt->il + 1;
    const rtype abstol = 0.0;
    const lapack_int i = LapackFn().call(
      opt->matrix_layout, opt->jobz, opt->range, opt->uplo, n, a, lda,
      opt->vl, opt->vu, opt->il, opt->iu, abstol, m, w, z, ldz, isuppz);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_heevr(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];
    VALUE nary_rtype = NaryTypes[nary_rtype_id];

    VALUE a_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);
    ID kw_table[8] = { rb_intern("jobz"), rb_intern("range"), rb_intern("uplo"),
                       rb_intern("vl"), rb_intern("vu"), rb_intern("il"), rb_intern("iu"), rb_intern("order") };
    VALUE kw_values[8] = { Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 8, kw_values);
    const char jobz = kw_values[0] != Qundef ? Util().get_jobz(kw_values[0]) : 'V';
    const char range = kw_values[1] != Qundef ? Util().get_range(kw_values[1]) : 'A';
    const char uplo = kw_values[2] != Qundef ? Util().get_uplo(kw_values[2]) : 'U';
    const rtype vl = kw_values[3] != Qundef ? NUM2DBL(kw_values[3]) : 0.0;
    const rtype vu = kw_values[4] != Qundef ? NUM2DBL(kw_values[4]) : 0.0;
    const lapack_int il = kw_values[5] != Qundef ? NUM2INT(kw_values[5]) : 0;
    const lapack_int iu = kw_values[6] != Qundef ? NUM2INT(kw_values[6]) : 0;
    const int matrix_layout = kw_values[7] != Qundef ? Util().get_matrix_layout(kw_values[7]) : LAPACK_ROW_MAJOR;

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
    size_t m = range != 'I' ? n : iu - il + 1;
    size_t w_shape[1] = { m };
    size_t z_shape[2] = { n, m };
    size_t isuppz_shape[1] = { 2 * m };
    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[5] = { { numo_cInt32, 0 }, { nary_rtype, 1, w_shape }, { nary_dtype, 2, z_shape }, { numo_cInt32, 1, isuppz_shape }, { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_heevr, NO_LOOP | NDF_EXTRACT, 1, 5, ain, aout };
    heevr_opt opt = { matrix_layout, jobz, range, uplo, vl, vu, il, iu };
    VALUE res = na_ndloop3(&ndf, &opt, 1, a_vnary);
    VALUE ret = rb_ary_new3(6, a_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1), rb_ary_entry(res, 2),
                            rb_ary_entry(res, 3), rb_ary_entry(res, 4));

    RB_GC_GUARD(a_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
