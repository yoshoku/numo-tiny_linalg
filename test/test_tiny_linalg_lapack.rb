# frozen_string_literal: true

require 'test_helper'

class TestTinyLinalgLapack < Minitest::Test # rubocop:disable Metrics/ClassLength
  def setup
    Numo::NArray.srand(53_196)
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

  def test_lapack_dpotrf
    n = 3
    a = Numo::DFloat.new(n, n).rand - 0.5
    b = a.transpose.dot(a)
    c, _info = Numo::TinyLinalg::Lapack.dpotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.dot(cu)).abs.max

    assert(error < 1e-7)

    c, _info = Numo::TinyLinalg::Lapack.dpotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_spotrf
    n = 3
    a = Numo::SFloat.new(n, n).rand - 0.5
    b = a.transpose.dot(a)
    c, _info = Numo::TinyLinalg::Lapack.spotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.dot(cu)).abs.max

    assert(error < 1e-5)

    c, _info = Numo::TinyLinalg::Lapack.spotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zpotrf
    n = 3
    a = Numo::DComplex.new(n, n).rand - 0.5
    b = a.transpose.conjugate.dot(a)
    c, _info = Numo::TinyLinalg::Lapack.zpotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.conjugate.dot(cu)).abs.max

    assert(error < 1e-7)

    c, _info = Numo::TinyLinalg::Lapack.zpotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose.conjugate)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cpotrf
    n = 3
    a = Numo::SComplex.new(n, n).rand - 0.5
    b = a.transpose.conjugate.dot(a)
    c, _info = Numo::TinyLinalg::Lapack.cpotrf(b.dup)
    cu = c.triu
    error = (b - cu.transpose.conjugate.dot(cu)).abs.max

    assert(error < 1e-5)

    c, _info = Numo::TinyLinalg::Lapack.cpotrf(b.dup, uplo: 'L')
    cl = c.tril
    error = (b - cl.dot(cl.transpose.conjugate)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dpotrs
    n = 5
    a = Numo::DFloat.new(n, n).rand
    c = a.transpose.dot(a)
    b = Numo::DFloat.new(n).rand - 0.5
    f, = Numo::TinyLinalg::Lapack.dpotrf(c.dup)
    x, _info = Numo::TinyLinalg::Lapack.dpotrs(f, b.dup)
    error = (b - c.dot(x)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_spotrs
    n = 5
    a = Numo::DFloat.new(n, n).rand
    c = a.transpose.dot(a)
    b = Numo::DFloat.new(n).rand - 0.5
    f, = Numo::TinyLinalg::Lapack.spotrf(c.dup, uplo: 'L')
    x, _info = Numo::TinyLinalg::Lapack.spotrs(f, b.dup, uplo: 'L')
    error = (b - c.dot(x)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zpotrs
    n = 5
    a = Numo::DComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::DComplex.new(n).rand - 0.5
    f, = Numo::TinyLinalg::Lapack.zpotrf(c.dup, uplo: 'L')
    x, _info = Numo::TinyLinalg::Lapack.zpotrs(f, b.dup, uplo: 'L')
    error = (b - c.dot(x)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cpotrs
    n = 5
    a = Numo::SComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    b = Numo::SComplex.new(n).rand - 0.5
    f, = Numo::TinyLinalg::Lapack.cpotrf(c.dup)
    x, _info = Numo::TinyLinalg::Lapack.cpotrs(f, b.dup)
    error = (b - c.dot(x)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dsyev
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::TinyLinalg::Lapack.dsyev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_ssyev
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::TinyLinalg::Lapack.ssyev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zheev
    n = 5
    a = Numo::DComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::TinyLinalg::Lapack.zheev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cheev
    n = 5
    a = Numo::SComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::TinyLinalg::Lapack.cheev(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dsyevd
    n = 5
    a = Numo::DFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::TinyLinalg::Lapack.dsyevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_ssyevd
    n = 5
    a = Numo::SFloat.new(n, n).rand - 0.5
    c = 0.5 * (a.transpose + a)
    v, w, _info = Numo::TinyLinalg::Lapack.ssyevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_zheevd
    n = 5
    a = Numo::DComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::TinyLinalg::Lapack.zheevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-7)
  end

  def test_lapack_cheevd
    n = 5
    a = Numo::SComplex.new(n, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    v, w, _info = Numo::TinyLinalg::Lapack.cheevd(c.dup, jobz: 'V', uplo: 'U')
    error = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error < 1e-5)
  end

  def test_lapack_dsyevr
    m = 3
    n = 5
    a = Numo::DFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    _c, _mm, w, v, _isuppz, _info = Numo::TinyLinalg::Lapack.dsyevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _c, mi, w, v, _isuppz, _info = Numo::TinyLinalg::Lapack.dsyevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _c, mv, w, v, _isuppz, _info = Numo::TinyLinalg::Lapack.dsyevr(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error_a < 1e-7)
    assert(error_i < 1e-7)
    # assert(error_v < 1e-7)
    assert(mi < n)
    # assert(mv < n)
  end

  def test_lapack_ssyevr
    m = 3
    n = 5
    a = Numo::SFloat.new(m, n).rand - 0.5
    c = a.transpose.dot(a)
    _c, _mm, w, v, _isuppz, _info = Numo::TinyLinalg::Lapack.ssyevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    _c, mi, w, v, _isuppz, _info = Numo::TinyLinalg::Lapack.ssyevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose)).abs.max
    # _c, mv, w, v, _isuppz, _info = Numo::TinyLinalg::Lapack.ssyevr(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose)).abs.max

    assert(error_a < 1e-5)
    assert(error_i < 1e-5)
    # assert(error_v < 1e-5)
    assert(mi < n)
    # assert(mv < n)
  end

  def test_lapack_zheevr
    m = 3
    n = 5
    a = Numo::DComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    _c, _mm, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.zheevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _c, mi, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.zheevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _c, mv, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.zheevr(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error_a < 1e-7)
    assert(error_i < 1e-7)
    # assert(error_v < 1e-7)
    assert(mi < n)
    # assert(mv < n)
  end

  def test_lapack_cheevr
    m = 3
    n = 5
    a = Numo::SComplex.new(m, n).rand - 0.5
    c = a.transpose.conjugate.dot(a)
    _c, _mm, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.cheevr(c.dup)
    error_a = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    _c, mi, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.cheevr(c.dup, range: 'I', il: 3, iu: 5)
    error_i = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max
    # _c, mv, w, v, _ifail, _info = Numo::TinyLinalg::Lapack.chegvx(c.dup, range: 'V', vl: 1e-6, vu: 1e+6)
    # error_v = (c - v.dot(w.diag).dot(v.transpose.conjugate)).abs.max

    assert(error_a < 1e-5)
    assert(error_i < 1e-5)
    # assert(error_v < 1e-5)
    assert(mi < n)
    # assert(mv < n)
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
