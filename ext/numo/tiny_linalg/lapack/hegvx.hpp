namespace TinyLinalg {

struct ZHeGvx {
  lapack_int call(int matrix_layout, lapack_int itype, char jobz, char range, char uplo,
                  lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_complex_double* b, lapack_int ldb,
                  double vl, double vu, lapack_int il, lapack_int iu,
                  double abstol, lapack_int* m, double* w, lapack_complex_double* z, lapack_int ldz, lapack_int* ifail) {
    return LAPACKE_zhegvx(matrix_layout, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, ifail);
  }
};

struct CHeGvx {
  lapack_int call(int matrix_layout, lapack_int itype, char jobz, char range, char uplo,
                  lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb,
                  float vl, float vu, lapack_int il, lapack_int iu,
                  float abstol, lapack_int* m, float* w, lapack_complex_float* z, lapack_int ldz, lapack_int* ifail) {
    return LAPACKE_chegvx(matrix_layout, itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, m, w, z, ldz, ifail);
  }
};

template <int nary_dtype_id, int nary_rtype_id, typename DType, typename RType, typename FncType>
class HeGvx {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_hegvx), -1);
  }

private:
  struct hegvx_opt {
    int matrix_layout;
    lapack_int itype;
    char jobz;
    char range;
    char uplo;
    RType vl;
    RType vu;
    lapack_int il;
    lapack_int iu;
  };

  static void iter_hegvx(na_loop_t* const lp) {
    DType* a = (DType*)NDL_PTR(lp, 0);
    DType* b = (DType*)NDL_PTR(lp, 1);
    int* m = (int*)NDL_PTR(lp, 2);
    RType* w = (RType*)NDL_PTR(lp, 3);
    DType* z = (DType*)NDL_PTR(lp, 4);
    int* ifail = (int*)NDL_PTR(lp, 5);
    int* info = (int*)NDL_PTR(lp, 6);
    hegvx_opt* opt = (hegvx_opt*)(lp->opt_ptr);
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int lda = NDL_SHAPE(lp, 0)[0];
    const lapack_int ldb = NDL_SHAPE(lp, 1)[0];
    const lapack_int ldz = opt->range != 'I' ? n : opt->iu - opt->il + 1;
    const RType abstol = 0.0;
    const lapack_int i = FncType().call(
      opt->matrix_layout, opt->itype, opt->jobz, opt->range, opt->uplo, n, a, lda, b, ldb,
      opt->vl, opt->vu, opt->il, opt->iu, abstol, m, w, z, ldz, ifail);
    *info = static_cast<int>(i);
  }

  static VALUE tiny_linalg_hegvx(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];
    VALUE nary_rtype = NaryTypes[nary_rtype_id];

    VALUE a_vnary = Qnil;
    VALUE b_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "2:", &a_vnary, &b_vnary, &kw_args);
    ID kw_table[9] = { rb_intern("itype"), rb_intern("jobz"), rb_intern("range"), rb_intern("uplo"),
                       rb_intern("vl"), rb_intern("vu"), rb_intern("il"), rb_intern("iu"), rb_intern("order") };
    VALUE kw_values[9] = { Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 9, kw_values);
    const lapack_int itype = kw_values[0] != Qundef ? get_itype(kw_values[0]) : 1;
    const char jobz = kw_values[1] != Qundef ? get_jobz(kw_values[1]) : 'V';
    const char range = kw_values[2] != Qundef ? get_range(kw_values[2]) : 'A';
    const char uplo = kw_values[3] != Qundef ? get_uplo(kw_values[3]) : 'U';
    const RType vl = kw_values[4] != Qundef ? NUM2DBL(kw_values[4]) : 0.0;
    const RType vu = kw_values[5] != Qundef ? NUM2DBL(kw_values[5]) : 0.0;
    const lapack_int il = kw_values[6] != Qundef ? NUM2INT(kw_values[6]) : 0;
    const lapack_int iu = kw_values[7] != Qundef ? NUM2INT(kw_values[7]) : 0;
    const int matrix_layout = kw_values[8] != Qundef ? get_matrix_layout(kw_values[8]) : LAPACK_ROW_MAJOR;

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
    GetNArray(a_vnary, b_nary);
    if (NA_NDIM(b_nary) != 2) {
      rb_raise(rb_eArgError, "input array b must be 2-dimensional");
      return Qnil;
    }
    if (NA_SHAPE(b_nary)[0] != NA_SHAPE(b_nary)[1]) {
      rb_raise(rb_eArgError, "input array b must be square");
      return Qnil;
    }

    const size_t n = NA_SHAPE(a_nary)[1];
    size_t m = range != 'I' ? n : iu - il + 1;
    size_t w_shape[1] = { m };
    size_t z_shape[2] = { n, m };
    size_t ifail_shape[1] = { n };
    ndfunc_arg_in_t ain[2] = { { OVERWRITE, 2 }, { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[5] = { { numo_cInt32, 0 }, { nary_rtype, 1, w_shape }, { nary_dtype, 2, z_shape }, { numo_cInt32, 1, ifail_shape }, { numo_cInt32, 0 } };
    ndfunc_t ndf = { iter_hegvx, NO_LOOP | NDF_EXTRACT, 2, 5, ain, aout };
    hegvx_opt opt = { matrix_layout, itype, jobz, range, uplo, vl, vu, il, iu };
    VALUE res = na_ndloop3(&ndf, &opt, 2, a_vnary, b_vnary);
    VALUE ret = rb_ary_new3(7, a_vnary, b_vnary, rb_ary_entry(res, 0), rb_ary_entry(res, 1), rb_ary_entry(res, 2),
                            rb_ary_entry(res, 3), rb_ary_entry(res, 4));

    RB_GC_GUARD(a_vnary);
    RB_GC_GUARD(b_vnary);

    return ret;
  }

  static lapack_int get_itype(VALUE val) {
    const lapack_int itype = NUM2INT(val);

    if (itype != 1 && itype != 2 && itype != 3) {
      rb_raise(rb_eArgError, "itype must be 1, 2 or 3");
    }

    return itype;
  }

  static char get_jobz(VALUE val) {
    const char jobz = NUM2CHR(val);

    if (jobz != 'N' && jobz != 'V') {
      rb_raise(rb_eArgError, "jobz must be 'N' or 'V'");
    }

    return jobz;
  }

  static char get_range(VALUE val) {
    const char range = NUM2CHR(val);

    if (range != 'A' && range != 'V' && range != 'I') {
      rb_raise(rb_eArgError, "range must be 'A', 'V' or 'I'");
    }

    return range;
  }

  static char get_uplo(VALUE val) {
    const char uplo = NUM2CHR(val);

    if (uplo != 'U' && uplo != 'L') {
      rb_raise(rb_eArgError, "uplo must be 'U' or 'L'");
    }

    return uplo;
  }

  static int get_matrix_layout(VALUE val) {
    const char option = NUM2CHR(val);

    switch (option) {
    case 'r':
    case 'R':
      break;
    case 'c':
    case 'C':
      rb_warn("Numo::TinyLinalg::Lapack.hegvx does not support column major.");
      break;
    }

    return LAPACK_ROW_MAJOR;
  }
};

} // namespace TinyLinalg
