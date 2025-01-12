# frozen_string_literal: true

require 'mkmf'
require 'numo/narray'
require 'open-uri'
require 'etc'
require 'fileutils'
require 'open3'
require 'digest/md5'
require 'rubygems/package'

$LOAD_PATH.each do |lp|
  if File.exist?(File.join(lp, 'numo/numo/narray.h'))
    $INCFLAGS = "-I#{lp}/numo #{$INCFLAGS}"
    break
  end
end

abort 'numo/narray.h is not found' unless have_header('numo/narray.h')

on_windows = RUBY_PLATFORM.match?(/mswin|cygwin|mingw/)

if on_windows
  $LOAD_PATH.each do |lp|
    if File.exist?(File.join(lp, 'numo/libnarray.a'))
      $LDFLAGS = "-L#{lp}/numo #{$LDFLAGS}"
      break
    end
  end
  abort 'libnarray.a is not found' unless have_library('narray', 'nary_new')
end

abort 'libstdc++ is not found.' unless have_library('stdc++')

build_openblas = false
unless find_library('openblas', 'LAPACKE_dsyevr')
  build_openblas = true unless have_library('openblas')
  build_openblas = true unless have_library('lapacke')
end
build_openblas = true unless have_header('cblas.h')
build_openblas = true unless have_header('lapacke.h')
build_openblas = true unless have_header('openblas_config.h')

if build_openblas
  warn 'BLAS and LAPACKE APIs are not found. Downloading and Building OpenBLAS...'

  VENDOR_DIR = File.expand_path("#{__dir__}/../../../vendor")
  TINYLINALG_DIR = File.expand_path("#{__dir__}/../../../lib/numo/tiny_linalg")
  OPENBLAS_VER = '0.3.29'
  OPENBLAS_KEY = '853a0c5c0747c5943e7ef4bbb793162d'
  OPENBLAS_URI = "https://github.com/OpenMathLib/OpenBLAS/archive/v#{OPENBLAS_VER}.tar.gz"
  OPENBLAS_TGZ = "#{VENDOR_DIR}/tmp/openblas.tgz"

  unless File.exist?("#{VENDOR_DIR}/installed_#{OPENBLAS_VER}")
    URI.parse(OPENBLAS_URI).open { |f| File.binwrite(OPENBLAS_TGZ, f.read) }
    abort('MD5 digest of downloaded OpenBLAS does not match.') if Digest::MD5.file(OPENBLAS_TGZ).to_s != OPENBLAS_KEY

    Gem::Package::TarReader.new(Zlib::GzipReader.open(OPENBLAS_TGZ)) do |tar|
      tar.each do |entry|
        next unless entry.file?

        filename = "#{VENDOR_DIR}/tmp/#{entry.full_name}"
        next if filename == File.dirname(filename)

        FileUtils.mkdir_p("#{VENDOR_DIR}/tmp/#{File.dirname(entry.full_name)}")
        File.binwrite(filename, entry.read)
        File.chmod(entry.header.mode, filename)
      end
    end

    Dir.chdir("#{VENDOR_DIR}/tmp/OpenBLAS-#{OPENBLAS_VER}") do
      mkstdout, _mkstderr, mkstatus = Open3.capture3("make -j#{Etc.nprocessors}")
      File.open("#{VENDOR_DIR}/tmp/openblas.log", 'w') { |f| f.puts(mkstdout) }
      abort("Failed to build OpenBLAS. Check the openblas.log file for more details: #{VENDOR_DIR}/tmp/openblas.log") unless mkstatus.success?

      insstdout, _insstderr, insstatus = Open3.capture3("make install PREFIX=#{VENDOR_DIR}")
      File.open("#{VENDOR_DIR}/tmp/openblas.log", 'a') { |f| f.puts(insstdout) }
      abort("Failed to install OpenBLAS. Check the openblas.log file for more details: #{VENDOR_DIR}/tmp/openblas.log") unless insstatus.success?

      FileUtils.cp("#{VENDOR_DIR}/lib/libopenblas.a", TINYLINALG_DIR) if on_windows

      FileUtils.touch("#{VENDOR_DIR}/installed_#{OPENBLAS_VER}")
    end
  end

  abort('libopenblas is not found.') unless find_library('openblas', nil, "#{VENDOR_DIR}/lib")
  abort('openblas_config.h is not found.') unless find_header('openblas_config.h', nil, "#{VENDOR_DIR}/include")
  abort('cblas.h is not found.') unless find_header('cblas.h', nil, "#{VENDOR_DIR}/include")
  abort('lapacke.h is not found.') unless find_header('lapacke.h', nil, "#{VENDOR_DIR}/include")
end

if RUBY_PLATFORM.include?('darwin') && Gem::Version.new('3.1.0') <= Gem::Version.new(RUBY_VERSION) &&
   try_link('int main(void){return 0;}', '-Wl,-undefined,dynamic_lookup')
  $LDFLAGS << ' -Wl,-undefined,dynamic_lookup'
end

create_makefile('numo/tiny_linalg/tiny_linalg')
