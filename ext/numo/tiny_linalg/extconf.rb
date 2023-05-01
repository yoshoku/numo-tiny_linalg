# frozen_string_literal: true

require 'mkmf'

use_accelerate = false
if RUBY_PLATFORM.include?('darwin') && have_framework('Accelerate')
  $CFLAGS << ' -DTINYLINALG_USE_ACCELERATE'
  use_accelerate = true
end

unless use_accelerate
  if have_library('openblas')
    $CFLAGS << ' -DTINYLINALG_USE_OPENBLAS'
  else
    abort 'libblas is not found' unless have_library('blas')
    $CFLAGS << ' -DTINYLINALG_USE_BLAS'
  end

  abort 'liblapack is not found' if !have_func('dsyevr') && !have_library('lapack')
end

create_makefile('numo/tiny_linalg/tiny_linalg')
