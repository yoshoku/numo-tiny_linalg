namespace TinyLinalg {
struct DConverter {
  double to_dtype(VALUE val) {
    return NUM2DBL(val);
  }

  double one() {
    return 1.0;
  }

  double zero() {
    return 0.0;
  }
};

struct SConverter {
  float to_dtype(VALUE val) {
    return static_cast<float>(NUM2DBL(val));
  }

  float one() {
    return 1.0f;
  }

  float zero() {
    return 0.0f;
  }
};

struct ZConverter {
  dcomplex to_dtype(VALUE val) {
    dcomplex z;
    REAL(z) = NUM2DBL(rb_funcall(val, rb_intern("real"), 0));
    IMAG(z) = NUM2DBL(rb_funcall(val, rb_intern("imag"), 0));
    return z;
  }

  dcomplex one() {
    dcomplex z;
    REAL(z) = 1.0;
    IMAG(z) = 0.0;
    return z;
  }

  dcomplex zero() {
    dcomplex z;
    REAL(z) = 0.0;
    IMAG(z) = 0.0;
    return z;
  }
};

struct CConverter {
  scomplex to_dtype(VALUE val) {
    scomplex z;
    REAL(z) = NUM2DBL(rb_funcall(val, rb_intern("real"), 0));
    IMAG(z) = NUM2DBL(rb_funcall(val, rb_intern("imag"), 0));
    return z;
  }

  scomplex one() {
    scomplex z;
    REAL(z) = 1.0f;
    IMAG(z) = 0.0f;
    return z;
  }

  scomplex zero() {
    scomplex z;
    REAL(z) = 0.0f;
    IMAG(z) = 0.0f;
    return z;
  }
};
} // namespace TinyLinalg
