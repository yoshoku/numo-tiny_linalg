#include "tiny_linalg.hpp"

VALUE rb_mTinyLinalg;

extern "C" void Init_tiny_linalg(void) {
  rb_mTinyLinalg = rb_define_module("TinyLinalg");
}
