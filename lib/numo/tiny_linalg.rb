# frozen_string_literal: true

if defined?(Numo::Linalg)
  warn 'Numo::Linalg is already defined. Numo::TinyLinalg will not be loaded.'
else
  require 'numo/narray'
  require_relative 'tiny_linalg/version'
  require_relative 'tiny_linalg/tiny_linalg'

  module Numo
    # Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
    module TinyLinalg
      module_function

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

    # Linalg = TinyLinalg
  end
end
