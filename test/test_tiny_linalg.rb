# frozen_string_literal: true

require 'test_helper'

class TestTinyLinalg < Minitest::Test
  def test_that_it_has_a_version_number
    refute_nil ::Numo::TinyLinalg::VERSION
  end

  def test_lapack_dgesvd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.dgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_sgesvd
    x = Numo::SFloat.new(5, 3).rand.dot(Numo::SFloat.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.sgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zgesvd
    x = Numo::DComplex.new(5, 3).rand.dot(Numo::DComplex.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.zgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cgesvd
    x = Numo::SComplex.new(5, 3).rand.dot(Numo::SComplex.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.cgesvd(x.dup, jobu: 'S', jobvt: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dgesdd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.dgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_sgesdd
    x = Numo::SFloat.new(5, 3).rand.dot(Numo::SFloat.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.sgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zgesdd
    x = Numo::DComplex.new(5, 3).rand.dot(Numo::DComplex.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.zgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cgesdd
    x = Numo::SComplex.new(5, 3).rand.dot(Numo::SComplex.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg::Lapack.cgesdd(x.dup, jobz: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-5)
  end

  def test_svd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg.svd(x.dup, driver: 'sdd', job: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert(error < 1e-7)
  end

  def test_blas_ddot
    x = Numo::DFloat[1, 2, 3]
    y = Numo::DFloat[4, 5, 6]

    assert_equal 32, Numo::TinyLinalg::Blas.ddot(x, y)
  end

  def test_blas_sdot
    x = Numo::SFloat[1, 2, 3]
    y = Numo::SFloat[4, 5, 6]

    assert_equal 32, Numo::TinyLinalg::Blas.sdot(x, y)
  end

  def test_blas_zdotu
    x = Numo::DComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)]
    y = Numo::DComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]

    assert_equal Complex(18, 43), Numo::TinyLinalg::Blas.zdotu(x, y)
  end

  def test_blas_cdotu
    x = Numo::SComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)]
    y = Numo::SComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]

    assert_equal Complex(18, 43), Numo::TinyLinalg::Blas.cdotu(x, y)
  end

  def test_blas_dgemm
    a = Numo::DFloat[[1, 2, 3], [4, 5, 6]]
    b = Numo::DFloat[[7, 8, 9], [3, 2, 1]]

    assert_equal Numo::DFloat[[19, 16, 13], [29, 26, 23], [39, 36, 33]], Numo::TinyLinalg::Blas.dgemm(a, b, transa: 'T')
    assert_equal Numo::DFloat[[50, 10], [122, 28]], Numo::TinyLinalg::Blas.dgemm(a, b, transb: 'T')
  end

  def test_blas_sgemm
    a = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    b = Numo::SFloat[[7, 8, 9], [3, 2, 1]]

    assert_equal Numo::SFloat[[19, 16, 13], [29, 26, 23], [39, 36, 33]], Numo::TinyLinalg::Blas.sgemm(a, b, transa: 'T')
    assert_equal Numo::SFloat[[50, 10], [122, 28]], Numo::TinyLinalg::Blas.sgemm(a, b, transb: 'T')
  end
end
