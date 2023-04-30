#include "tiny_linalg.h"

VALUE rb_mTinyLinalg;

void
Init_tiny_linalg(void)
{
  rb_mTinyLinalg = rb_define_module("TinyLinalg");
}
