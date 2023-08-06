# frozen_string_literal: true

require 'test_helper'

class TestTinyLinalgLapack < Minitest::Test # rubocop:disable Metrics/ClassLength, Style/Documentation
  def setup
    Numo::NArray.srand(Minitest.seed)
  end

  def test_lapack_dgeqrf_dorgqr
    ma = 3
    na = 2
    a = Numo::DFloat.new(ma, na).rand
    qr, tau, = Numo::TinyLinalg::Lapack.dgeqrf(a.dup)
    r = qr.triu
    qq = Numo::DFloat.zeros(ma, ma)
    qq[0...ma, 0...na] = qr
    q, = Numo::TinyLinalg::Lapack.dorgqr(qq, tau)
    error_a = (a - q.dot(r)).abs.max

    mb = 2
    nb = 3
    b = Numo::DFloat.new(mb, nb).rand
    qr, tau, = Numo::TinyLinalg::Lapack.dgeqrf(b.dup)
    r = qr.triu
    q, = Numo::TinyLinalg::Lapack.dorgqr(qr[true, 0...mb], tau)
    error_b = (b - q.dot(r)).abs.max

    assert(error_a < 1e-7)
    assert(error_b < 1e-7)
  end

  def test_lapack_sgeqrf_sorgqr
    ma = 3
    na = 2
    a = Numo::SFloat.new(ma, na).rand
    qr, tau, = Numo::TinyLinalg::Lapack.sgeqrf(a.dup)
    r = qr.triu
    qq = Numo::SFloat.zeros(ma, ma)
    qq[0...ma, 0...na] = qr
    q, = Numo::TinyLinalg::Lapack.sorgqr(qq, tau)
    error_a = (a - q.dot(r)).abs.max

    mb = 2
    nb = 3
    b = Numo::SFloat.new(mb, nb).rand
    qr, tau, = Numo::TinyLinalg::Lapack.sgeqrf(b.dup)
    r = qr.triu
    q, = Numo::TinyLinalg::Lapack.sorgqr(qr[true, 0...mb], tau)
    error_b = (b - q.dot(r)).abs.max

    assert(error_a < 1e-5)
    assert(error_b < 1e-5)
  end

  def test_lapack_zgeqrf_zungqr
    ma = 3
    na = 2
    a = Numo::DComplex.new(ma, na).rand
    qr, tau, = Numo::TinyLinalg::Lapack.zgeqrf(a.dup)
    r = qr.triu
    qq = Numo::DComplex.zeros(ma, ma)
    qq[0...ma, 0...na] = qr
    q, = Numo::TinyLinalg::Lapack.zungqr(qq, tau)
    error_a = (a - q.dot(r)).abs.max

    mb = 2
    nb = 3
    b = Numo::DComplex.new(mb, nb).rand
    qr, tau, = Numo::TinyLinalg::Lapack.zgeqrf(b.dup)
    r = qr.triu
    q, = Numo::TinyLinalg::Lapack.zungqr(qr[true, 0...mb], tau)
    error_b = (b - q.dot(r)).abs.max

    assert(error_a < 1e-7)
    assert(error_b < 1e-7)
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

  def test_lapack_dsygv
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::DFloat.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.dsygv(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_ssygv
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::SFloat.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.ssygv(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zhegv
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::DFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.zhegv(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_chegv
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::SFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.chegv(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dsygvd
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::DFloat.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.dsygvd(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_ssygvd
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    b = Numo::SFloat.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.ssygvd(c.dup, b.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zhegvd
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::DFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.zhegvd(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_chegvd
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    a = 0.5 * (a.transpose + a)
    b = Numo::SFloat.new(n, n).rand
    b = 0.5 * (b.transpose + b)
    b = (b.triu - b.tril)
    b[b.diag_indices] = 0.0
    c = a + (b * Complex::I)
    d = Numo::DComplex.eye(n)
    v, _x, w, _info = Numo::TinyLinalg::Lapack.chegvd(c.dup, d.dup, itype: 1, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dsygvx
    m = 3
    n = 5
    a = Numo::DFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    b = Numo::DFloat.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.dsygvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.dsygvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.dsygvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error_a < 1e-7)
    assert(error_i < 1e-7)
    # assert(error_v < 1e-7)
    assert(mi < n)
    # assert(mv < n)
  end

  def test_lapack_ssygvx
    m = 3
    n = 5
    a = Numo::SFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    b = Numo::SFloat.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.ssygvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.ssygvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.ssygvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error_a < 1e-5)
    assert(error_i < 1e-5)
    # assert(error_v < 1e-5)
    assert(mi < n)
    # assert(mv < n)
  end

  def test_lapack_zhegvx
    m = 3
    n = 5
    a = Numo::DComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::DComplex.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.zhegvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.zhegvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.zhegvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error_a < 1e-7)
    assert(error_i < 1e-7)
    # assert(error_v < 1e-7)
    assert(mi < n)
    # assert(mv < n)
  end

  def test_lapack_chegvx
    m = 3
    n = 5
    a = Numo::SComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::SComplex.eye(n)
    _a, _b, _mm, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.chegvx(c.dup, b.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _a, _b, mi, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.chegvx(c.dup, b.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _a, _b, mv, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.chegvx(c.dup, b.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error_a < 1e-5)
    assert(error_i < 1e-5)
    # assert(error_v < 1e-5)
    assert(mi < n)
    # assert(mv < n)
  end
end
