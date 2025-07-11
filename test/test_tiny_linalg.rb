# frozen_string_literal: true

require 'test_helper'

class TestTinyLinalg < Minitest::Test # rubocop:disable Metrics/ClassLength
  def setup
    Numo::NArray.srand(53_196)
  end

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

  def test_eigh
    m = 3
    n = 5
    x = Numo::DFloat.new(m, n).rand - 0.5
    a = x.transpose.dot(x)
    v, w = Numo::TinyLinalg.eigh(a)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    v, w = Numo::TinyLinalg.eigh(a, vals_range: (n - m)...n)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    x = Numo::DComplex.new(m, n).rand - 0.5
    a = x.transpose.conjugate.dot(x)
    v, w = Numo::TinyLinalg.eigh(a, turbo: true)
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    v, w = Numo::TinyLinalg.eigh(a, vals_range: [n - m, n - 1])
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)

    x = Numo::DFloat.new(m, n).rand - 0.5
    y = Numo::DFloat.new(m, n).rand - 0.5
    a = x.transpose.dot(x)
    b = y.transpose.dot(y) + (n * Numo::DFloat.eye(n))
    v, w = Numo::TinyLinalg.eigh(a, b)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)

    v, w = Numo::TinyLinalg.eigh(a, b, vals_range: (n - m)...n)
    r = w.transpose.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)

    x = Numo::DComplex.new(m, n).rand - 0.5
    y = Numo::DComplex.new(m, n).rand - 0.5
    a = x.transpose.conjugate.dot(x)
    b = y.transpose.conjugate.dot(y) + (n * Numo::DComplex.eye(n))
    v, w = Numo::TinyLinalg.eigh(a, b, turbo: true)
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.conjugate.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)

    v, w = Numo::TinyLinalg.eigh(a, b, vals_range: [n - m, n - 1])
    r = w.transpose.conjugate.dot(a.dot(w))
    r = r[r.diag_indices]
    e = w.transpose.conjugate.dot(b.dot(w))
    e = e[e.diag_indices]

    assert_operator((v - r).abs.max, :<, 1e-7)
    assert_operator((e - 1).abs.max, :<, 1e-7)
  end

  def test_norm
    # empty array
    assert_equal(0, Numo::TinyLinalg.norm([]))
    assert_equal(0, Numo::TinyLinalg.norm(Numo::DFloat[]))

    # vector
    a = Numo::DFloat[3, -4]
    b = Numo::DFloat[1, 0, 2, 0, 3]

    assert_equal(5, Numo::TinyLinalg.norm(a))
    assert_equal(5, Numo::TinyLinalg.norm(a, 2))
    assert_equal(7, Numo::TinyLinalg.norm(a, 1))
    assert_equal(3, Numo::TinyLinalg.norm(b, 0))
    assert_in_delta(2.4, Numo::TinyLinalg.norm(a, -2))
    assert_equal(4, Numo::TinyLinalg.norm(a, Float::INFINITY))
    assert_equal(3, Numo::TinyLinalg.norm(a, -Float::INFINITY))
    assert_equal(4, Numo::TinyLinalg.norm(a, 'inf'))
    assert_equal(3, Numo::TinyLinalg.norm(a, '-inf'))
    assert_equal(Numo::DFloat[5], Numo::TinyLinalg.norm(a, keepdims: true))
    assert_equal(Numo::DFloat[5], Numo::TinyLinalg.norm(a, 2, keepdims: true))
    assert_equal(5, Numo::TinyLinalg.norm(a, axis: 0))
    assert_match(/axis is out of range/, assert_raises(ArgumentError) { Numo::TinyLinalg.norm(a, axis: 1) }.message)
    assert_match(/invalid axis/, assert_raises(ArgumentError) { Numo::TinyLinalg.norm(a, axis: '1') }.message)

    # matrix
    a = Numo::DFloat[[1, 2, -3, 1], [-4, 1, 8, 2]]

    assert_equal(10, Numo::TinyLinalg.norm(a))
    assert_equal(10, Numo::TinyLinalg.norm(a, 'fro'))
    assert_operator((Numo::TinyLinalg.norm(a, 'nuc') - 12.3643).abs, :<, 1e-4)
    assert_operator((Numo::TinyLinalg.norm(a, 2) - 9.6144).abs, :<, 1e-4)
    assert_equal(11, Numo::TinyLinalg.norm(a, 1))
    assert_match(/invalid ord/, assert_raises(ArgumentError) { Numo::TinyLinalg.norm(a, 0) }.message)
    assert_equal(3, Numo::TinyLinalg.norm(a, -1))
    assert_operator((Numo::TinyLinalg.norm(a, -2) - 2.7498).abs, :<, 1e-4)
    assert_equal(15, Numo::TinyLinalg.norm(a, Float::INFINITY))
    assert_equal(7, Numo::TinyLinalg.norm(a, -Float::INFINITY))
    assert_equal(15, Numo::TinyLinalg.norm(a, 'inf'))
    assert_equal(7, Numo::TinyLinalg.norm(a, '-inf'))
    assert_equal(Numo::DFloat[[10]], Numo::TinyLinalg.norm(a, keepdims: true))
    assert_equal(Numo::DFloat[5, 3, 11, 3], Numo::TinyLinalg.norm(a, 1, axis: 0))
    assert_equal(Numo::DFloat[7, 15], Numo::TinyLinalg.norm(a, 1, axis: 1))
    assert_equal(Numo::DFloat[[5, 3, 11, 3]], Numo::TinyLinalg.norm(a, 1, axis: 0, keepdims: true))
    assert_equal(Numo::DFloat[[7], [15]], Numo::TinyLinalg.norm(a, 1, axis: 1, keepdims: true))
    assert_equal(10, Numo::TinyLinalg.norm(a, 'fro', axis: [0, 1]))
    assert_equal(11, Numo::TinyLinalg.norm(a, 1, axis: [0, 1]))
    assert_equal(Numo::DFloat[[15]], Numo::TinyLinalg.norm(a, Float::INFINITY, axis: [0, 1], keepdims: true))

    # tensor
    a = Numo::DFloat[[[2, 3, 1], [1, 2, 4]], [[2, 2, 3], [3, 2, 4]]]

    assert_equal(9, Numo::TinyLinalg.norm(a))
    assert_equal(Numo::DFloat[[[9]]], Numo::TinyLinalg.norm(a, keepdims: true))
  end

  def test_cholesky
    a = Numo::DFloat.new(3, 3).rand - 0.5
    b = a.transpose.dot(a)
    u = Numo::TinyLinalg.cholesky(b)
    error = (b - u.transpose.dot(u)).abs.max

    assert_operator(error, :<, 1e-7)

    a = Numo::SComplex.new(3, 3).rand - 0.5
    b = a.transpose.conjugate.dot(a)
    l = Numo::TinyLinalg.cholesky(b, uplo: 'L')
    error = (b - l.dot(l.transpose.conjugate)).abs.max

    assert_operator(error, :<, 1e-5)
  end

  def test_cho_solve
    a = Numo::DFloat.new(3, 3).rand - 0.5
    c = a.transpose.dot(a)
    u = Numo::TinyLinalg.cholesky(c)
    b = Numo::DFloat.new(3, 2).rand - 0.5
    x = Numo::TinyLinalg.cho_solve(u, b)
    error = (b - c.dot(x)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_det
    a = Numo::DFloat[[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    error = (Numo::TinyLinalg.det(a) - 3).abs

    assert_operator(error, :<, 1e-7)
  end

  def test_inv
    a = Numo::DFloat.new(3, 3).rand - 0.5
    a_inv = Numo::TinyLinalg.inv(a)
    error = (Numo::DFloat.eye(3) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_pinv
    a = Numo::DComplex.new(5, 3).rand - 0.5
    a_inv = Numo::TinyLinalg.pinv(a)
    error = (Numo::DComplex.eye(3) - a_inv.dot(a)).abs.max

    assert_operator(error, :<, 1e-7)
  end

  def test_qr
    ma = 5
    na = 3
    a = Numo::DFloat.new(ma, na).rand - 0.5
    q, r = Numo::TinyLinalg.qr(a, mode: 'economic')
    error_a = (a - q.dot(r)).abs.max

    mb = 3
    nb = 5
    b = Numo::DFloat.new(mb, nb).rand - 0.5
    q, r = Numo::TinyLinalg.qr(b, mode: 'economic')
    error_b = (b - q.dot(r)).abs.max

    mc = 5
    nc = 3
    c = Numo::DComplex.new(mc, nc).rand - 0.5
    q, r = Numo::TinyLinalg.qr(c, mode: 'economic')
    error_c = (c - q.dot(r)).abs.max

    md = 3
    nd = 5
    d = Numo::DComplex.new(md, nd).rand - 0.5
    q, r = Numo::TinyLinalg.qr(d, mode: 'economic')
    error_d = (d - q.dot(r)).abs.max

    q, r = Numo::TinyLinalg.qr(a, mode: 'reduce')

    assert_operator(error_a, :<, 1e-7)
    assert_operator(error_b, :<, 1e-7)
    assert_operator(error_c, :<, 1e-7)
    assert_operator(error_d, :<, 1e-7)
    assert_equal(q.shape, [ma, ma])
    assert_equal(r.shape, [ma, na])
  end

  def test_solve
    a = Numo::DComplex.new(3, 3).rand
    b = Numo::SFloat.new(3).rand
    x = Numo::TinyLinalg.solve(a, b)
    error_ab = (b - a.dot(x)).abs.max

    assert_operator(error_ab, :<, 1e-7)
  end

  def test_solve_triangular
    a = Numo::SFloat.new(3, 3).rand.triu
    b = Numo::DComplex.new(3).rand
    x = Numo::TinyLinalg.solve_triangular(a, b)
    error_ab = (b - a.dot(x)).abs.max

    assert_operator(error_ab, :<, 1e-7)
  end

  def test_svd
    x = Numo::DFloat.new(5, 3).rand.dot(Numo::DFloat.new(3, 2).rand)
    s, u, vt, = Numo::TinyLinalg.svd(x, driver: 'sdd', job: 'S')
    z = u.dot(s.diag).dot(vt)
    error = (x - z).abs.max

    assert_operator(error, :<, 1e-7)
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

    assert_operator(error_ab, :<, 1e-7)
    assert_operator(error_ac, :<, 1e-7)
    assert_operator(error_cb, :<, 1e-7)
    assert_operator(error_cd, :<, 1e-7)
  end
end
