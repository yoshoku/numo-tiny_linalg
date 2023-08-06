namespace TinyLinalg {

struct DGeSdd {
  lapack_int call(int matrix_order, char jobz, lapack_int m, lapack_int n,
                  double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt) {
    return LAPACKE_dgesdd(matrix_order, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
  };
};

struct SGeSdd {
  lapack_int call(int matrix_order, char jobz, lapack_int m, lapack_int n,
                  float* a, lapack_int lda, float* s, float* u, lapack_int ldu, float* vt, lapack_int ldvt) {
    return LAPACKE_sgesdd(matrix_order, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
  };
};

struct ZGeSdd {
  lapack_int call(int matrix_order, char jobz, lapack_int m, lapack_int n,
                  lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt) {
    return LAPACKE_zgesdd(matrix_order, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
  };
};

struct CGeSdd {
  lapack_int call(int matrix_order, char jobz, lapack_int m, lapack_int n,
                  lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt) {
    return LAPACKE_cgesdd(matrix_order, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
  };
};

template <int nary_dtype_id, int nary_rtype_id, typename dtype, typename rtype, class LapackFn>
class GeSdd {
public:
  static void define_module_function(VALUE mLapack, const char* mf_name) {
    rb_define_module_function(mLapack, mf_name, RUBY_METHOD_FUNC(tiny_linalg_gesdd), -1);
  };

private:
  struct gesdd_opt {
    int matrix_order;
    char jobz;
  };

  static void iter_gesdd(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    rtype* s = (rtype*)NDL_PTR(lp, 1);
    dtype* u = (dtype*)NDL_PTR(lp, 2);
    dtype* vt = (dtype*)NDL_PTR(lp, 3);
    int* info = (int*)NDL_PTR(lp, 4);
    gesdd_opt* opt = (gesdd_opt*)(lp->opt_ptr);

    const size_t m = opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[0] : NDL_SHAPE(lp, 0)[1];
    const size_t n = opt->matrix_order == LAPACK_ROW_MAJOR ? NDL_SHAPE(lp, 0)[1] : NDL_SHAPE(lp, 0)[0];
    const size_t min_mn = m < n ? m : n;
    const lapack_int lda = n;
    const lapack_int ldu = opt->jobz == 'S' ? min_mn : m;
    const lapack_int ldvt = opt->jobz == 'S' ? min_mn : n;

    lapack_int i = LapackFn().call(opt->matrix_order, opt->jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
    *info = static_cast<int>(i);
  };

  static VALUE tiny_linalg_gesdd(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];
    VALUE nary_rtype = NaryTypes[nary_rtype_id];
    VALUE a_vnary = Qnil;
    VALUE kw_args = Qnil;

    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);

    ID kw_table[2] = { rb_intern("jobz"), rb_intern("order") };
    VALUE kw_values[2] = { Qundef, Qundef };

    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);

    const char jobz = kw_values[0] == Qundef ? 'A' : StringValueCStr(kw_values[0])[0];
    const char order = kw_values[1] == Qundef ? 'R' : StringValueCStr(kw_values[1])[0];

    if (CLASS_OF(a_vnary) != nary_dtype) {
      rb_raise(rb_eTypeError, "type of input array is invalid for overwriting");
      return Qnil;
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

    switch (jobz) {
    case 'A':
      break;
    case 'S':
      shape_u[matrix_order == LAPACK_ROW_MAJOR ? 1 : 0] = min_mn;
      shape_vt[matrix_order == LAPACK_ROW_MAJOR ? 0 : 1] = min_mn;
      break;
    case 'O':
      break;
    case 'N':
      aout[1].dim = 0;
      aout[2].dim = 0;
      break;
    default:
      rb_raise(rb_eArgError, "jobz must be one of 'A', 'S', 'O', or 'N'");
      return Qnil;
    }

    ndfunc_t ndf = { iter_gesdd, NO_LOOP | NDF_EXTRACT, 1, 4, ain, aout };
    gesdd_opt opt = { matrix_order, jobz };
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);

    RB_GC_GUARD(a_vnary);
    return ret;
  };
};

} // namespace TinyLinalg
