# frozen_string_literal: true

require 'test_helper'

class TestTinyLinalg < Minitest::Test # rubocop:disable Metrics/ClassLength
  def test_that_it_has_a_version_number
    refute_nil ::Numo::TinyLinalg::VERSION
  end

  def test_blas_char
    assert_equal 'd', Numo::TinyLinalg.blas_char([true, false])
    assert_equal 'd', Numo::TinyLinalg.blas_char([1, 2])
    assert_equal 'd', Numo::TinyLinalg.blas_char([1.1, 2.2])
    assert_equal 'z', Numo::TinyLinalg.blas_char([Complex(1, 2), 3])
    assert_equal 'd', Numo::TinyLinalg.blas_char(Numo::NArray[1, 2])
    assert_equal 's', Numo::TinyLinalg.blas_char(Numo::SFloat[1.1, 2.2])
    assert_equal 'd', Numo::TinyLinalg.blas_char(Numo::DFloat[1.1, 2.2])
    assert_equal 'c', Numo::TinyLinalg.blas_char(Numo::SComplex[1.1, 2.2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::DComplex[1.1, 2.2])
    assert_equal 'd', Numo::TinyLinalg.blas_char(Numo::SFloat[1, 2], Numo::DFloat[1, 2])
    assert_equal 'c', Numo::TinyLinalg.blas_char(Numo::SFloat[1, 2], Numo::SComplex[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::SFloat[1, 2], Numo::DComplex[1, 2])
    assert_equal 'd', Numo::TinyLinalg.blas_char(Numo::DFloat[1, 2], Numo::SFloat[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::DFloat[1, 2], Numo::SComplex[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::DFloat[1, 2], Numo::DComplex[1, 2])
    assert_equal 'c', Numo::TinyLinalg.blas_char(Numo::SComplex[1, 2], Numo::SFloat[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::SComplex[1, 2], Numo::DFloat[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::SComplex[1, 2], Numo::DComplex[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::DComplex[1, 2], Numo::SFloat[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::DComplex[1, 2], Numo::DFloat[1, 2])
    assert_equal 'z', Numo::TinyLinalg.blas_char(Numo::DComplex[1, 2], Numo::SComplex[1, 2])
    assert_raises TypeError, 'invalid data type for BLAS/LAPACK' do
      Numo::TinyLinalg.blas_char(['1', 2, 3])
    end
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

    x = (Numo::DFloat.new(3, 2).rand * 10).floor
    y = (Numo::DFloat.new(2, 5).rand * 10).floor

    assert_equal x.dot(y), Numo::TinyLinalg::Blas.dgemm(x, y)
  end

  def test_blas_sgemm
    a = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    b = Numo::SFloat[[7, 8, 9], [3, 2, 1]]

    assert_equal Numo::SFloat[[19, 16, 13], [29, 26, 23], [39, 36, 33]], Numo::TinyLinalg::Blas.sgemm(a, b, transa: 'T')
    assert_equal Numo::SFloat[[50, 10], [122, 28]], Numo::TinyLinalg::Blas.sgemm(a, b, transb: 'T')
  end

  def test_blas_zgemm
    a = (Numo::DComplex.new(2, 3).rand * 10).floor
    b = (Numo::DComplex.new(2, 3).rand * 10).floor

    assert_equal a.dot(b.transpose), Numo::TinyLinalg::Blas.zgemm(a, b, transb: 'T')
    assert_equal a.transpose.dot(b), Numo::TinyLinalg::Blas.zgemm(a, b, transa: 'T')
  end

  def test_blas_cgemm
    a = (Numo::SComplex.new(2, 3).rand * 10).floor
    b = (Numo::SComplex.new(2, 3).rand * 10).floor

    assert_equal a.dot(b.transpose), Numo::TinyLinalg::Blas.cgemm(a, b, transb: 'T')
    assert_equal a.transpose.dot(b), Numo::TinyLinalg::Blas.cgemm(a, b, transa: 'T')
  end

  def test_blas_dgemv
    a = Numo::DFloat[[1, 2, 3], [4, 5, 6]]
    x = Numo::DFloat[7, 8, 9]
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::TinyLinalg::Blas.dgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::TinyLinalg::Blas.dgemv(b, x, trans: 'T')
  end

  def test_blas_sgemv
    a = Numo::SFloat[[1, 2, 3], [4, 5, 6]]
    x = Numo::SFloat[7, 8, 9]
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::TinyLinalg::Blas.sgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::TinyLinalg::Blas.sgemv(b, x, trans: 'T')
  end

  def test_blas_zgemv
    a = (Numo::DComplex.new(2, 3).rand * 10).floor
    x = (Numo::DComplex.new(3).rand * 10).floor
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::TinyLinalg::Blas.zgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::TinyLinalg::Blas.zgemv(b, x, trans: 'T')
  end

  def test_blas_cgemv
    a = (Numo::SComplex.new(2, 3).rand * 10).floor
    x = (Numo::SComplex.new(3).rand * 10).floor
    b = a.transpose.dup

    assert_equal a.dot(x), Numo::TinyLinalg::Blas.cgemv(a, x)
    assert_equal b.transpose.dot(x), Numo::TinyLinalg::Blas.cgemv(b, x, trans: 'T')
  end

  def test_blas_dnrm2
    a = Numo::DFloat.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a)) - Numo::TinyLinalg::Blas.dnrm2(a)).abs.max

    assert(error < 1e-7)
  end

  def test_blas_snrm2
    a = Numo::SFloat.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a)) - Numo::TinyLinalg::Blas.dnrm2(a)).abs.max

    assert(error < 1e-5)
  end
end
