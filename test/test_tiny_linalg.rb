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

  def test_lapack_dgesv
    a = Numo::DFloat.new(5, 5).rand
    b = Numo::DFloat.new(5).rand
    c = Numo::DFloat.new(5, 5).rand
    d = Numo::DFloat.new(5, 3).rand
    ret_ab = Numo::TinyLinalg::Lapack.dgesv(a.dup, b.dup)
    ret_cd = Numo::TinyLinalg::Lapack.dgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert(error_ab < 1e-7)
    assert(error_cd < 1e-7)
  end

  def test_lapack_sgesv
    a = Numo::SFloat.new(3, 3).rand
    b = Numo::SFloat.new(3).rand
    c = Numo::SFloat.new(3, 3).rand
    d = Numo::SFloat.new(3, 5).rand
    ret_ab = Numo::TinyLinalg::Lapack.sgesv(a.dup, b.dup)
    ret_cd = Numo::TinyLinalg::Lapack.sgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert(error_ab < 1e-5)
    assert(error_cd < 1e-5)
  end

  def test_lapack_zgesv
    a = Numo::DComplex.new(5, 5).rand
    b = Numo::DComplex.new(5).rand
    c = Numo::DComplex.new(5, 5).rand
    d = Numo::DComplex.new(5, 3).rand
    ret_ab = Numo::TinyLinalg::Lapack.zgesv(a.dup, b.dup)
    ret_cd = Numo::TinyLinalg::Lapack.zgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert(error_ab < 1e-7)
    assert(error_cd < 1e-7)
  end

  def test_lapack_cgesv
    a = Numo::SComplex.new(3, 3).rand
    b = Numo::SComplex.new(3).rand
    c = Numo::SComplex.new(3, 3).rand
    d = Numo::SComplex.new(3, 5).rand
    ret_ab = Numo::TinyLinalg::Lapack.cgesv(a.dup, b.dup)
    ret_cd = Numo::TinyLinalg::Lapack.cgesv(c.dup, d.dup)
    error_ab = (b - a.dot(ret_ab[1])).abs.max
    error_cd = (d - c.dot(ret_cd[1])).abs.max

    assert(error_ab < 1e-5)
    assert(error_cd < 1e-5)
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

  def test_lapack_dgetrf
    nr = 3
    nc = 2
    a = Numo::DFloat.new(nr, nc).rand
    lu, piv, = Numo::TinyLinalg::Lapack.dgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }
    u = lu.triu[0...nc, 0...nc]
    pm = Numo::DFloat.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_sgetrf
    nr = 3
    nc = 5
    a = Numo::SFloat.new(nr, nc).rand
    lu, piv, = Numo::TinyLinalg::Lapack.sgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }[0...nr, 0...nr]
    u = lu.triu[0...nr, 0...nc]
    pm = Numo::SFloat.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zgetrf
    nr = 3
    nc = 2
    a = Numo::DComplex.new(nr, nc).rand
    lu, piv, = Numo::TinyLinalg::Lapack.zgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }
    u = lu.triu[0...nc, 0...nc]
    pm = Numo::DComplex.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cgetrf
    nr = 3
    nc = 2
    a = Numo::SComplex.new(nr, nc).rand
    lu, piv, = Numo::TinyLinalg::Lapack.cgetrf(a.dup)
    l = lu.tril.tap { |m| m[m.diag_indices] = 1 }
    u = lu.triu[0...nc, 0...nc]
    pm = Numo::SComplex.eye(nr).tap { |m| piv.each_with_index { |v, i| m[true, [v - 1, i]] = m[true, [i, v - 1]].dup } }
    error = (a - pm.dot(l).dot(u)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dgetri
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    lu, piv, = Numo::TinyLinalg::Lapack.dgetrf(a.dup)
    a_inv, = Numo::TinyLinalg::Lapack.dgetri(lu, piv)
    error = (Numo::DFloat.eye(n) - a_inv.dot(a)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_sgetri
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    lu, piv, = Numo::TinyLinalg::Lapack.sgetrf(a.dup)
    a_inv, = Numo::TinyLinalg::Lapack.sgetri(lu, piv)
    error = (Numo::SFloat.eye(n) - a_inv.dot(a)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zgetri
    n = 3
    a = Numo::DComplex.new(n, n).rand
    lu, piv, = Numo::TinyLinalg::Lapack.zgetrf(a.dup)
    a_inv, = Numo::TinyLinalg::Lapack.zgetri(lu, piv)
    error = (Numo::DComplex.eye(n) - a_inv.dot(a)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cgetri
    n = 3
    a = Numo::SComplex.new(n, n).rand
    lu, piv, = Numo::TinyLinalg::Lapack.cgetrf(a.dup)
    a_inv, = Numo::TinyLinalg::Lapack.cgetri(lu, piv)
    error = (Numo::SComplex.eye(n) - a_inv.dot(a)).abs.max

    assert(error < 1e-5)
  end

  def test_solve
    a = Numo::DComplex.new(3, 3).rand
    b = Numo::SFloat.new(3).rand
    x = Numo::TinyLinalg.solve(a, b)
    error_ab = (b - a.dot(x)).abs.max

    assert(error_ab < 1e-7)
  end

  def test_svd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg.svd(x, driver: 'sdd', job: 'S')
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
    error = (Numo::NMath.sqrt(a.dot(a)) - Numo::TinyLinalg::Blas.snrm2(a)).abs.max

    assert(error < 1e-5)
  end

  def test_blas_dznrm2
    a = Numo::DComplex.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a.conjugate)) - Numo::TinyLinalg::Blas.dznrm2(a)).abs.max

    assert(error < 1e-7)
  end

  def test_blas_scnrm2
    a = Numo::SComplex.new(3).rand
    error = (Numo::NMath.sqrt(a.dot(a.conjugate)) - Numo::TinyLinalg::Blas.scnrm2(a)).abs.max

    assert(error < 1e-5)
  end

  def test_blas_aliases
    a = Numo::DComplex.new(3).rand
    b = Numo::SComplex.new(3).rand
    error_a = (Numo::NMath.sqrt(a.dot(a.conjugate)) - Numo::TinyLinalg::Blas.znrm2(a)).abs.max
    error_b = (Numo::NMath.sqrt(b.dot(b.conjugate)) - Numo::TinyLinalg::Blas.cnrm2(b)).abs.max

    assert(error_a < 1e-7)
    assert(error_b < 1e-5)
  end

  def test_blas_call
    assert_equal 32, Numo::TinyLinalg::Blas.call(:dot, Numo::DFloat[1, 2, 3], Numo::DFloat[4, 5, 6])
    assert_equal 32, Numo::TinyLinalg::Blas.call(:dot, Numo::SFloat[1, 2, 3], Numo::SFloat[4, 5, 6])
    assert_equal Complex(18, 43), Numo::TinyLinalg::Blas.call(
      :dotu,
      Numo::DComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)],
      Numo::DComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]
    )
    assert_equal Complex(18, 43), Numo::TinyLinalg::Blas.call(
      :dotu,
      Numo::SComplex[Complex(1, 0), Complex(2, 1), Complex(3, 2)],
      Numo::SComplex[Complex(4, 3), Complex(5, 4), Complex(6, 5)]
    )
  end

  def test_dot
    a = Numo::DFloat.new(3).rand
    b = Numo::SFloat.new(3).rand
    c = Numo::DFloat.new(3, 2).rand
    d = Numo::SFloat.new(2, 3).rand
    error_ab = (a.dot(b) - Numo::TinyLinalg.dot(a, b)).abs
    error_ac = (a.dot(c) - Numo::TinyLinalg.dot(a, c)).abs.max
    error_cb = (c.transpose.dot(b) - Numo::TinyLinalg.dot(c.transpose, b)).abs.max
    error_cd = (c.dot(d) - Numo::TinyLinalg.dot(c, d)).abs.max

    assert(error_ab < 1e-7)
    assert(error_ac < 1e-7)
    assert(error_cb < 1e-7)
    assert(error_cd < 1e-7)
  end
end
