# frozen_string_literal: true

require 'mkmf'
require 'numo/narray'

$LOAD_PATH.each do |lp|
  if File.exist?(File.join(lp, 'numo/numo/narray.h'))
    $INCFLAGS = "-I#{lp}/numo #{$INCFLAGS}"
    break
  end
end

abort 'numo/narray.h is not found' unless have_header('numo/narray.h')

if RUBY_PLATFORM.match?(/mswin|cygwin|mingw/)
  $LOAD_PATH.each do |lp|
    if File.exist?(File.join(lp, 'numo/libnarray.a'))
      $LDFLAGS = "-L#{lp}/numo #{$LDFLAGS}"
      break
    end
  end
  abort 'libnarray.a is not found' unless have_library('narray', 'nary_new')
end

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
