# frozen_string_literal: true

require 'numo/narray'
require_relative 'tiny_linalg/version'
require_relative 'tiny_linalg/tiny_linalg'

# Ruby/Numo (NUmerical MOdules)
module Numo
  # Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
  module TinyLinalg
    module_function

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
