namespace TinyLinalg {

struct DSyGvx {
  lapack_int call(int matrix_layout, lapack_int itype, char jobz, char range, char uplo,
                  lapack_int n, double* a, lapack_int lda, double* b, lapack_int ldb,
                  double vl, double vu, lapack_int il, lapack_int iu,
                  double abstol, lapack_int* m, double* w, double* z, lapack_int ldz, lapack_int* ifail) {
    return LAPACKE_dsygvx(matrix_layout, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, ifail);
  }
};

struct SSyGvx {
  lapack_int call(int matrix_layout, lapack_int itype, char jobz, char range, char uplo,
                  lapack_int n, float* a, lapack_int lda, float* b, lapack_int ldb,
                  float vl, float vu, lapack_int il, lapack_int iu,
                  float abstol, lapack_int* m, float* w, float* z, lapack_int ldz, lapack_int* ifail) {
    return LAPACKE_ssygvx(matrix_layout, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, ifail);
  }
};

template <int nary_dtype_id, typename dtype, class LapackFn>
class SyGvx {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_sygvx), -1);
  }

private:
  struct sygvx_opt {
    int matrix_layout;
    lapack_int itype;
    char jobz;
    char range;
    char uplo;
    dtype vl;
    dtype vu;
    lapack_int il;
    lapack_int iu;
  };

  static void iter_sygvx(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    dtype* b = (dtype*)NDL_PTR(lp, 1);
    int* m = (int*)NDL_PTR(lp, 2);
    dtype* w = (dtype*)NDL_PTR(lp, 3);
    dtype* z = (dtype*)NDL_PTR(lp, 4);
    int* ifail = (int*)NDL_PTR(lp, 5);
    int* info = (int*)NDL_PTR(lp, 6);
    sygvx_opt* opt = (sygvx_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int lda = NDL_SHAPE(lp, 0)[0];
    const lapack_int ldb = NDL_SHAPE(lp, 1)[0];
    const lapack_int ldz = opt->range != 'I' ? n : opt->iu - opt->il + 1;
    const dtype abstol = 0.0;
    const lapack_int i = LapackFn().call(
      opt->matrix_layout, opt->itype, opt->jobz, opt->range, opt->uplo, n, a, lda, b, ldb,
      opt->vl, opt->vu, opt->il, opt->iu, abstol, m, w, z, ldz, ifail);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_sygvx(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE b_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);
    ID kw_table[9] = { rb_intern("itype"), rb_intern("jobz"), rb_intern("range"), rb_intern("uplo"),
                       rb_intern("vl"), rb_intern("vu"), rb_intern("il"), rb_intern("iu"), rb_intern("order") };
    VALUE kw_values[9] = { Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 9, kw_values);
    const lapack_int itype = kw_values[0] != Qundef ? Util().get_itype(kw_values[0]) : 1;
    const char jobz = kw_values[1] != Qundef ? Util().get_jobz(kw_values[1]) : 'V';
    const char range = kw_values[2] != Qundef ? Util().get_range(kw_values[2]) : 'A';
    const char uplo = kw_values[3] != Qundef ? Util().get_uplo(kw_values[3]) : 'U';
    const dtype vl = kw_values[4] != Qundef ? NUM2DBL(kw_values[4]) : 0.0;
    const dtype vu = kw_values[5] != Qundef ? NUM2DBL(kw_values[5]) : 0.0;
    const lapack_int il = kw_values[6] != Qundef ? NUM2INT(kw_values[6]) : 0;
    const lapack_int iu = kw_values[7] != Qundef ? NUM2INT(kw_values[7]) : 0;
    const int matrix_layout = kw_values[8] != Qundef ? Util().get_matrix_layout(kw_values[8]) : LAPACK_ROW_MAJOR;

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

    if (range == 'V' && vu <= vl) {
      rb_raise(rb_eArgError, "vu must be greater than vl");
      return Qnil;
    }

    const size_t n = NA_SHAPE(a_nary)[1];
    if (range == 'I' && (il < 1 || il > n)) {
      rb_raise(rb_eArgError, "il must satisfy 1 <= il <= n");
      return Qnil;
    }
    if (range == 'I' && (iu < 1 || iu > n)) {
      rb_raise(rb_eArgError, "iu must satisfy 1 <= iu <= n");
      return Qnil;
    }
    if (range == 'I' && iu < il) {
      rb_raise(rb_eArgError, "iu must be greater than or equal to il");
      return Qnil;
    }

    size_t m = range != 'I' ? n : iu - il + 1;
    size_t w_shape[1] = { m };
    size_t z_shape[2] = { n, m };
    size_t ifail_shape[1] = { n };
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[5] = { { numo_cInt32, 0 }, { nary_dtype, 1, w_shape }, { nary_dtype, 2, z_shape }, { numo_cInt32, 1, ifail_shape }, { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_sygvx, NO_LOOP | NDF_EXTRACT, 2, 5, ain, aout };
    sygvx_opt opt = { matrix_layout, itype, jobz, range, uplo, vl, vu, il, iu };
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);
    VALUE ret = rb_ary_new3(7, a_vnary, b_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1), rb_ary_entry(res, 2),
                            rb_ary_entry(res, 3), rb_ary_entry(res, 4));

    RB_GC_GUARD(a_vnary);
    RB_GC_GUARD(b_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
