namespace TinyLinalg {

struct DLanGe {
  double call(int matrix_layout, char norm, lapack_int m, lapack_int n, const double* a, lapack_int lda) {
    return LAPACKE_dlange(matrix_layout, norm, m, n, a, lda);
  }
};

struct SLanGe {
  float call(int matrix_layout, char norm, lapack_int m, lapack_int n, const float* a, lapack_int lda) {
    return LAPACKE_slange(matrix_layout, norm, m, n, a, lda);
  }
};

struct ZLanGe {
  double call(int matrix_layout, char norm, lapack_int m, lapack_int n, const lapack_complex_double* a, lapack_int lda) {
    return LAPACKE_zlange(matrix_layout, norm, m, n, a, lda);
  }
};

struct CLanGe {
  float call(int matrix_layout, char norm, lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda) {
    return LAPACKE_clange(matrix_layout, norm, m, n, a, lda);
  }
};

template <int nary_dtype_id, typename dtype, class LapackFn>
class LanGe {
public:
  static void define_module_function(VALUE mLapack, const char* fnc_name) {
    rb_define_module_function(mLapack, fnc_name, RUBY_METHOD_FUNC(tiny_linalg_lange), -1);
  }

private:
  struct lange_opt {
    int matrix_layout;
    char norm;
  };

  static void iter_lange(na_loop_t* const lp) {
    dtype* a = (dtype*)NDL_PTR(lp, 0);
    dtype* d = (dtype*)NDL_PTR(lp, 1);
    lange_opt* opt = (lange_opt*)(lp->opt_ptr);
    const lapack_int m = NDL_SHAPE(lp, 0)[0];
    const lapack_int n = NDL_SHAPE(lp, 0)[1];
    const lapack_int lda = n;
    *d = LapackFn().call(opt->matrix_layout, opt->norm, m, n, a, lda);
  }

  static VALUE tiny_linalg_lange(int argc, VALUE* argv, VALUE self) {
    VALUE nary_dtype = NaryTypes[nary_dtype_id];

    VALUE a_vnary = Qnil;
    VALUE kw_args = Qnil;
    rb_scan_args(argc, argv, "1:", &a_vnary, &kw_args);
    ID kw_table[2] = { rb_intern("order"), rb_intern("norm") };
    VALUE kw_values[2] = { Qundef, Qundef };
    rb_get_kwargs(kw_args, kw_table, 0, 2, kw_values);
    const int matrix_layout = kw_values[0] != Qundef ? Util().get_matrix_layout(kw_values[0]) : LAPACK_ROW_MAJOR;
    const char norm = kw_values[1] != Qundef ? NUM2CHR(kw_values[1]) : 'F';

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

    ndfunc_arg_in_t ain[1] = { { nary_dtype, 2 } };
    size_t shape_out[1] = { 1 };
    ndfunc_arg_out_t aout[1] = { { nary_dtype, 0, shape_out } };
    ndfunc_t ndf = { iter_lange, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };
    lange_opt opt = { matrix_layout, norm };
    VALUE ret = na_ndloop3(&ndf, &opt, 1, a_vnary);

    RB_GC_GUARD(a_vnary);

    return ret;
  }
};

} // namespace TinyLinalg
