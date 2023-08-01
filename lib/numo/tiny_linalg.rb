# frozen_string_literal: true

require 'numo/narray'
require_relative 'tiny_linalg/version'
require_relative 'tiny_linalg/tiny_linalg'

# Ruby/Numo (NUmerical MOdules)
module Numo
  # Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
  module TinyLinalg # rubocop:disable Metrics/ModuleLength
    module_function

    # Computes the determinant of matrix.
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
