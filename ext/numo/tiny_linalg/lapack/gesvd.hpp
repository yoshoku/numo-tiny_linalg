namespace TinyLinalg {

struct DGeSvd {
  lapack_int call(int matrix_order, char jobu, char jobvt, lapack_int m, lapack_int n,
                  double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt,
                  double* superb) {
    return LAPACKE_dgesvd(matrix_order, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
  };
};

struct SGeSvd {
  lapack_int call(int matrix_order, char jobu, char jobvt, lapack_int m, lapack_int n,
                  float* a, lapack_int lda, float* s, float* u, lapack_int ldu, float* vt, lapack_int ldvt,
                  float* superb) {
    return LAPACKE_sgesvd(matrix_order, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
  };
};

struct ZGeSvd {
  lapack_int call(int matrix_order, char jobu, char jobvt, lapack_int m, lapack_int n,
                  lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt,
                  double* superb) {
    return LAPACKE_zgesvd(matrix_order, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
  };
};

struct CGeSvd {
  lapack_int call(int matrix_order, char jobu, char jobvt, lapack_int m, lapack_int n,
                  lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt,
                  float* superb) {
    return LAPACKE_cgesvd(matrix_order, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
  };
};

template <int nary_dtype_id, int nary_rtype_id, typename dtype, typename rtype, class LapackFn>
class GeSvd {
public:
  static void define_module_function(VALUE mLapack, const char* mf_name) {
    rb_define_module_function(mLapack, mf_name, RUBY_METHOD_FUNC(tiny_linalg_gesvd), -1);
  };

private:
  struct gesvd_opt {
    int matrix_order;
    char jobu;
    char jobvt;
  };

  static void iter_gesvd(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    rtype* s = (rtype*)NDL_PTR(lp, 1);
    dtype* u = (dtype*)NDL_PTR(lp, 2);
    dtype* vt = (dtype*)NDL_PTR(lp, 3);
    int* info = (int*)NDL_PTR(lp, 4);
    gesvd_opt* opt = (gesvd_opt*)(lp->opt_ptr);

    const size_t m = opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0] : NDL_SHAPE(lp, 0)[1];
    const size_t n = opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[1] : NDL_SHAPE(lp, 0)[0];
    const size_t min_mn = m < n ? m : n;
    const lapack_int lda = n;
    const lapack_int ldu = opt->jobu == 'A' ? m : min_mn;
    const lapack_int ldvt = n;

    rtype* superb = (rtype*)ruby_xmalloc(min_mn * sizeof(rtype));

    lapack_int i = LapackFn().call(opt->matrix_order, opt->jobu, opt->jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
    *info = static_cast<int>(i);

    ruby_xfree(superb);
  };

  static VALUE tiny_linalg_gesvd(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];
    VALUE nary_rtype = NaryTypes[nary_rtype_id];
    VALUE a_vnary = Qnil;
    VALUE kw_args = Qnil;

    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);

    ID kw_table[3] = { rb_intern("jobu"), rb_intern("jobvt"), rb_intern("order") };
    VALUE kw_values[3] = { Qundef, Qundef, Qundef };

    rb_get_kwargs(kw_args, kw_table, 0, 3, kw_values);

    const char jobu = kw_values[0] == Qundef ? 'A' : StringValueCStr(kw_values[0])[0];
    const char jobvt = kw_values[1] == Qundef ? 'A' : StringValueCStr(kw_values[1])[0];
    const char order = kw_values[2] == Qundef ? 'R' : StringValueCStr(kw_values[2])[0];

    if (jobu == 'O' && jobvt == 'O') {
      rb_raise(rb_eArgError, "jobu and jobvt cannot be both 'O'");
      return Qnil;
    }
    if (CLASS_OF(a_vnary) != nary_dtype) {
      rb_raise(rb_eTypeError, "type of input array is invalid for overwriting");
      return Qnil;
    }

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
      rb_raise(rb_eArgError, "input array must be 2-dimensional");
      return Qnil;
    }

    const int matrix_order = order == 'C' ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
    const size_t m = matrix_order == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[0] : NA_SHAPE(a_nary)[1];
    const size_t n = matrix_order == LAPACK_ROW_MAJOR ? NA_SHAPE(a_nary)[1] : NA_SHAPE(a_nary)[0];

    const size_t min_mn = m < n ? m : n;
    size_t shape_s[1] = { min_mn };
    size_t shape_u[2] = { m, m };
    size_t shape_vt[2] = { n, n };

    ndfunc_arg_in_t ain[1] = { { OVERWRITE, 2 } };
    ndfunc_arg_out_t aout[4] = { { nary_rtype, 1, shape_s }, { nary_dtype, 2, shape_u }, { nary_dtype, 2, shape_vt }, { numo_cInt32, 0 } };

    switch (jobu) {
    case 'A':
      break;
    case 'S':
      shape_u[matrix_order == LAPACK_ROW_MAJOR ? 1 : 0] = min_mn;
      break;
    case 'O':
    case 'N':
      aout[1].dim = 0;
      break;
    default:
      rb_raise(rb_eArgError, "jobu must be 'A', 'S', 'O', or 'N'");
      return Qnil;
    }

    switch (jobvt) {
    case 'A':
      break;
    case 'S':
      shape_vt[matrix_order == LAPACK_ROW_MAJOR ? 0 : 1] = min_mn;
      break;
    case 'O':
    case 'N':
      aout[2].dim = 0;
      break;
    default:
      rb_raise(rb_eArgError, "jobvt must be 'A', 'S', 'O', or 'N'");
      return Qnil;
    }

    ndfunc_t ndf = { iter_gesvd, NO_LOOP | NDF_EXTRACT, 1, 4, ain, aout };
    gesvd_opt opt = { matrix_order, jobu, jobvt };
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);

    switch (jobu) {
    case 'O':
      rb_ary_store(ret, 1, a_vnary);
      break;
    case 'N':
      rb_ary_store(ret, 1, Qnil);
      break;
    }

    switch (jobvt) {
    case 'O':
      rb_ary_store(ret, 2, a_vnary);
      break;
    case 'N':
      rb_ary_store(ret, 2, Qnil);
      break;
    }

    RB_GC_GUARD(a_vnary);
    return ret;
  };
};

} // namespace TinyLinalg
