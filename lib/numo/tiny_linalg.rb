# frozen_string_literal: true

require 'numo/narray'
require_relative 'tiny_linalg/version'
require_relative 'tiny_linalg/tiny_linalg'

# Ruby/Numo (NUmerical MOdules)
module Numo
  # Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
  module TinyLinalg # rubocop:disable Metrics/ModuleLength
    module_function

    # Computes the eigenvalues and eigenvectors of a symmetric / Hermitian matrix
    # by solving an ordinary or generalized eigenvalue problem.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   x = Numo::DFloat.new(5, 3).rand - 0.5
    #   c = x.dot(x.transpose)
    #   vals, vecs = Numo::Linalg.eigh(c, vals_range: [2, 4])
    #
    #   pp vals
    #   # =>
    #   # Numo::DFloat#shape=[3]
    #   # [0.118795, 0.434252, 0.903245]
    #
    #   pp vecs
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[0.154178, 0.60661, -0.382961],
    #   #  [-0.349761, -0.141726, -0.513178],
    #   #  [0.739633, -0.468202, 0.105933],
    #   #  [0.0519655, -0.471436, -0.701507],
    #   #  [-0.551488, -0.412883, 0.294371]]
    #
    #   pp (x - vecs.dot(vals.diag).dot(vecs.transpose)).abs.max
    #   # => 3.3306690738754696e-16
    #
    # @param a [Numo::NArray] n-by-n symmetric / Hermitian matrix.
    # @param b [Numo::NArray] n-by-n symmetric / Hermitian matrix. If nil, identity matrix is assumed.
    # @param vals_only [Boolean] The flag indicating whether to return only eigenvalues.
    # @param vals_range [Range/Array]
    #   The range of indices of the eigenvalues (in ascending order) and corresponding eigenvectors to be returned.
    #   If nil, all eigenvalues and eigenvectors are computed.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param turbo [Bool] The flag indicating whether to use a divide and conquer algorithm. If vals_range is given, this flag is ignored.
    # @return [Array<Numo::NArray, Numo::NArray>] The eigenvalues and eigenvectors.
    def eigh(a, b = nil, vals_only: false, vals_range: nil, uplo: 'U', turbo: false) # rubocop:disable Metrics/AbcSize, Metrics/ParameterLists, Lint/UnusedMethodArgument
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      unless b.nil?
        raise ArgumentError, 'input array b must be 2-dimensional' if b.ndim != 2
        raise ArgumentError, 'input array b must be square' if b.shape[0] != b.shape[1]
        raise ArgumentError, "invalid array type: #{b.class}" if blas_char(b) == 'n'
      end

      jobz = vals_only ? 'N' : 'V'
      b = a.class.eye(a.shape[0]) if b.nil?
      sy_he_gv = %w[d s].include?(bchr) ? "#{bchr}sygv" : "#{bchr}hegv"

      if vals_range.nil?
        sy_he_gv << 'd' if turbo
        vecs, _b, vals, _info = Numo::TinyLinalg::Lapack.send(sy_he_gv.to_sym, a.dup, b.dup, jobz: jobz)
      else
        sy_he_gv << 'x'
        il = vals_range.first + 1
        iu = vals_range.last + 1
        _a, _b, _m, vals, vecs, _ifail, _info = Numo::TinyLinalg::Lapack.send(
          sy_he_gv.to_sym, a.dup, b.dup, jobz: jobz, range: 'I', il: il, iu: iu
        )
      end
      vecs = nil if vals_only
      [vals, vecs]
    end

    # Computes the determinant of matrix.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   a = Numo::DFloat[[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    #   pp (3.0 - Numo::Linalg.det(a)).abs
    #   # => 1.3322676295501878e-15
    #
    # @param a [Numo::NArray] n-by-n square matrix.
    # @return [Float/Complex] The determinant of `a`.
    def det(a)
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      getrf = "#{bchr}getrf".to_sym
      lu, piv, info = Numo::TinyLinalg::Lapack.send(getrf, a.dup)

      if info.zero?
        det_l = 1
        det_u = lu.diagonal.prod
        det_p = piv.map_with_index { |v, i| v == i + 1 ? 1 : -1 }.prod
        det_l * det_u * det_p
      elsif info.positive?
        raise 'the factor U is singular, and the inverse matrix could not be computed.'
      else
        raise "the #{-info}-th argument of getrf had illegal value"
      end
    end

    # Computes the inverse matrix of a square matrix.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   a = Numo::DFloat.new(5, 5).rand
    #
    #   inv_a = Numo::Linalg.inv(a)
    #
    #   pp (inv_a.dot(a) - Numo::DFloat.eye(5)).abs.max
    #   # => 7.019165976816745e-16
    #
    #   pp inv_a.dot(a).sum
    #   # => 5.0
    #
    # @param a [Numo::NArray] n-by-n square matrix.
    # @param driver [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @return [Numo::NArray] The inverse matrix of `a`.
    def inv(a, driver: 'getrf', uplo: 'U') # rubocop:disable Lint/UnusedMethodArgument
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      getrf = "#{bchr}getrf".to_sym
      getri = "#{bchr}getri".to_sym

      lu, piv, info = Numo::TinyLinalg::Lapack.send(getrf, a.dup)
      if info.zero?
        Numo::TinyLinalg::Lapack.send(getri, lu, piv)[0]
      elsif info.positive?
        raise 'the factor U is singular, and the inverse matrix could not be computed.'
      else
        raise "the #{-info}-th argument of getrf had illegal value"
      end
    end

    # Compute the (Moore-Penrose) pseudo-inverse of a matrix using singular value decomposition.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   a = Numo::DFloat.new(5, 3).rand
    #
    #   inv_a = Numo::Linalg.pinv(a)
    #
    #   pp (inv_a.dot(a) - Numo::DFloat.eye(3)).abs.max
    #   # => 1.1102230246251565e-15
    #
    #   pp inv_a.dot(a).sum
    #   # => 3.0
    #
    # @param a [Numo::NArray] The m-by-n matrix to be pseudo-inverted.
    # @param driver [String] LAPACK driver to be used ('svd' or 'sdd').
    # @param rcond [Float] The threshold value for small singular values of `a`, default value is `a.shape.max * EPS`.
    # @return [Numo::NArray] The pseudo-inverse of `a`.
    def pinv(a, driver: 'svd', rcond: nil)
      s, u, vh = svd(a, driver: driver, job: 'S')
      rcond = a.shape.max * s.class::EPSILON if rcond.nil?
      rank = s.gt(rcond * s[0]).count

      u = u[true, 0...rank] / s[0...rank]
      u.dot(vh[0...rank, true]).conj.transpose
    end

    # Compute QR decomposition of a matrix.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   x = Numo::DFloat.new(5, 3).rand
    #
    #   q, r = Numo::Linalg.qr(x, mode: 'economic')
    #
    #   pp q
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[-0.0574417, 0.635216, 0.707116],
    #   #  [-0.187002, -0.073192, 0.422088],
    #   #  [-0.502239, 0.634088, -0.537489],
    #   #  [-0.0473292, 0.134867, -0.0223491],
    #   #  [-0.840979, -0.413385, 0.180096]]
    #
    #   pp r
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[-1.07508, -0.821334, -0.484586],
    #   #  [0, 0.513035, 0.451868],
    #   #  [0, 0, 0.678737]]
    #
    #   pp (q.dot(r) - x).abs.max
    #   # => 3.885780586188048e-16
    #
    # @param a [Numo::NArray] The m-by-n matrix to be decomposed.
    # @param mode [String] The mode of decomposition.
    #   - "reduce"   -- returns both Q [m, m] and R [m, n],
    #   - "r"        -- returns only R,
    #   - "economic" -- returns both Q [m, n] and R [n, n],
    #   - "raw"      -- returns QR and TAU (LAPACK geqrf results).
    # @return [Numo::NArray] if mode='r'
    # @return [Array<Numo::NArray,Numo::NArray>] if mode='reduce' or mode='economic'
    # @return [Array<Numo::NArray,Numo::NArray>] if mode='raw' (LAPACK geqrf result)
    def qr(a, mode: 'reduce')
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, "invalid mode: #{mode}" unless %w[reduce r economic raw].include?(mode)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      geqrf = "#{bchr}geqrf".to_sym
      qr, tau, = Numo::TinyLinalg::Lapack.send(geqrf, a.dup)

      return [qr, tau] if mode == 'raw'

      m, n = qr.shape
      r = m > n && %w[economic raw].include?(mode) ? qr[0...n, true].triu : qr.triu

      return r if mode == 'r'

      org_ung_qr = %w[d s].include?(bchr) ? "#{bchr}orgqr".to_sym : "#{bchr}ungqr".to_sym

      q = if m < n
            Numo::TinyLinalg::Lapack.send(org_ung_qr, qr[true, 0...m], tau)[0]
          elsif mode == 'economic'
            Numo::TinyLinalg::Lapack.send(org_ung_qr, qr, tau)[0]
          else
            qqr = a.class.zeros(m, m)
            qqr[0...m, 0...n] = qr
            Numo::TinyLinalg::Lapack.send(org_ung_qr, qqr, tau)[0]
          end

      [q, r]
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` from square matrix `a`.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   a = Numo::DFloat.new(3, 3).rand
    #   b = Numo::DFloat.eye(3)
    #
    #   x = Numo::Linalg.solve(a, b)
    #
    #   pp x
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[-2.12332, 4.74868, 0.326773],
    #   #  [1.38043, -3.79074, 1.25355],
    #   #  [0.775187, 1.41032, -0.613774]]
    #
    #   pp (b - a.dot(x)).abs.max
    #   # => 2.1081041547796492e-16
    #
    # @param a [Numo::NArray] The n-by-n square matrix  (>= 2-dimensinal NArray).
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix (>= 1-dimensinal NArray).
    # @param driver [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @return [Numo::NArray] The solusion vector / matrix `x`.
    def solve(a, b, driver: 'gen', uplo: 'U') # rubocop:disable Lint/UnusedMethodArgument
      case blas_char(a, b)
      when 'd'
        Lapack.dgesv(a.dup, b.dup)[1]
      when 's'
        Lapack.sgesv(a.dup, b.dup)[1]
      when 'z'
        Lapack.zgesv(a.dup, b.dup)[1]
      when 'c'
        Lapack.cgesv(a.dup, b.dup)[1]
      end
    end

    # Calculates the Singular Value Decomposition (SVD) of a matrix: `A = U * S * V^T`
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   x = Numo::DFloat.new(5, 2).rand.dot(Numo::DFloat.new(2, 3).rand)
    #   pp x
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[0.104945, 0.0284236, 0.117406],
    #   #  [0.862634, 0.210945, 0.922135],
    #   #  [0.324507, 0.0752655, 0.339158],
    #   #  [0.67085, 0.102594, 0.600882],
    #   #  [0.404631, 0.116868, 0.46644]]
    #
    #   s, u, vt = Numo::Linalg.svd(x, job: 'S')
    #
    #   z = u.dot(s.diag).dot(vt)
    #   pp z
    #   # =>
    #   # Numo::DFloat#shape=[5,3]
    #   # [[0.104945, 0.0284236, 0.117406],
    #   #  [0.862634, 0.210945, 0.922135],
    #   #  [0.324507, 0.0752655, 0.339158],
    #   #  [0.67085, 0.102594, 0.600882],
    #   #  [0.404631, 0.116868, 0.46644]]
    #
    #   pp (x - z).abs.max
    #   # => 4.440892098500626e-16
    #
    # @param a [Numo::NArray] Matrix to be decomposed.
    # @param driver [String] LAPACK driver to be used ('svd' or 'sdd').
    # @param job [String] Job option ('A', 'S', or 'N').
    # @return [Array<Numo::NArray>] Singular values and singular vectors ([s, u, vt]).
    def svd(a, driver: 'svd', job: 'A')
      raise ArgumentError, "invalid job: #{job}" unless /^[ASN]/i.match?(job.to_s)

      case driver.to_s
      when 'sdd'
        s, u, vt, info = case a
                         when Numo::DFloat
                           Numo::TinyLinalg::Lapack.dgesdd(a.dup, jobz: job)
                         when Numo::SFloat
                           Numo::TinyLinalg::Lapack.sgesdd(a.dup, jobz: job)
                         when Numo::DComplex
                           Numo::TinyLinalg::Lapack.zgesdd(a.dup, jobz: job)
                         when Numo::SComplex
                           Numo::TinyLinalg::Lapack.cgesdd(a.dup, jobz: job)
                         else
                           raise ArgumentError, "invalid array type: #{a.class}"
                         end
      when 'svd'
        s, u, vt, info = case a
                         when Numo::DFloat
                           Numo::TinyLinalg::Lapack.dgesvd(a.dup, jobu: job, jobvt: job)
                         when Numo::SFloat
                           Numo::TinyLinalg::Lapack.sgesvd(a.dup, jobu: job, jobvt: job)
                         when Numo::DComplex
                           Numo::TinyLinalg::Lapack.zgesvd(a.dup, jobu: job, jobvt: job)
                         when Numo::SComplex
                           Numo::TinyLinalg::Lapack.cgesvd(a.dup, jobu: job, jobvt: job)
                         else
                           raise ArgumentError, "invalid array type: #{a.class}"
                         end
      else
        raise ArgumentError, "invalid driver: #{driver}"
      end

      raise "the #{info.abs}-th argument had illegal value" if info.negative?
      raise 'input array has a NAN entry' if info == -4
      raise 'svd did not converge' if info.positive?

      [s, u, vt]
    end
  end
end
