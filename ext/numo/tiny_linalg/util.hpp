namespace TinyLinalg {

class Util {
public:
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
      rb_warn("Numo::TinyLinalg does not support column major.");
      break;
    }

    return LAPACK_ROW_MAJOR;
  }

  static enum CBLAS_TRANSPOSE get_cblas_trans(VALUE val) {
    const char option = NUM2CHR(val);
    enum CBLAS_TRANSPOSE res = CblasNoTrans;

    switch (option) {
    case 'n':
    case 'N':
      res = CblasNoTrans;
      break;
    case 't':
    case 'T':
      res = CblasTrans;
      break;
    case 'c':
    case 'C':
      res = CblasConjTrans;
      break;
    }

    return res;
  }

  static enum CBLAS_ORDER get_cblas_order(VALUE val) {
    const char option = NUM2CHR(val);

    switch (option) {
    case 'r':
    case 'R':
      break;
    case 'c':
    case 'C':
      rb_warn("Numo::TinyLinalg does not support column major.");
      break;
    }

    return CblasRowMajor;
  }
};

} // namespace TinyLinalg
