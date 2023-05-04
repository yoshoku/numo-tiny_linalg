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

if RUBY_PLATFORM.include?('darwin') && Gem::Version.new('3.1.0') <= Gem::Version.new(RUBY_VERSION) &&
   try_link('int main(void){return 0;}', '-Wl,-undefined,dynamic_lookup')
  $LDFLAGS << ' -Wl,-undefined,dynamic_lookup'
end

use_accelerate = false
# NOTE: Accelerate framework does not support LAPACKE.
# if RUBY_PLATFORM.include?('darwin') && have_framework('Accelerate')
#   $CFLAGS << ' -DTINYLINALG_USE_ACCELERATE'
#   use_accelerate = true
# end

unless use_accelerate
  if have_library('openblas')
    $CFLAGS << ' -DTINYLINALG_USE_OPENBLAS'
  else
    abort 'libblas is not found' unless have_library('blas')
    $CFLAGS << ' -DTINYLINALG_USE_BLAS'
  end

  abort 'liblapack is not found' if !have_func('dsyevr') && !have_library('lapack')
  abort 'cblas.h is not found' unless have_header('cblas.h')
  abort 'lapacke.h is not found' unless have_header('lapacke.h')
end

abort 'libstdc++ is not found.' unless have_library('stdc++')

$CXXFLAGS << ' -std=c++11'

create_makefile('numo/tiny_linalg/tiny_linalg')
