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
    # @param a [Numo::NArray] The n-by-n symmetric / Hermitian matrix.
    # @param b [Numo::NArray] The n-by-n symmetric / Hermitian matrix. If nil, identity matrix is assumed.
    # @param vals_only [Boolean] The flag indicating whether to return only eigenvalues.
    # @param vals_range [Range/Array]
    #   The range of indices of the eigenvalues (in ascending order) and corresponding eigenvectors to be returned.
    #   If nil, all eigenvalues and eigenvectors are computed.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param turbo [Bool] The flag indicating whether to use a divide and conquer algorithm. If vals_range is given, this flag is ignored.
    # @return [Array<Numo::NArray>] The eigenvalues and eigenvectors.
    def eigh(a, b = nil, vals_only: false, vals_range: nil, uplo: 'U', turbo: false) # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/ParameterLists, Metrics/PerceivedComplexity, Lint/UnusedMethodArgument
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      b_given = !b.nil?
      raise ArgumentError, 'input array b must be 2-dimensional' if b_given && b.ndim != 2
      raise ArgumentError, 'input array b must be square' if b_given && b.shape[0] != b.shape[1]
      raise ArgumentError, "invalid array type: #{b.class}" if b_given && blas_char(b) == 'n'

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      jobz = vals_only ? 'N' : 'V'

      if b_given
        fnc = %w[d s].include?(bchr) ? "#{bchr}sygv" : "#{bchr}hegv"
        if vals_range.nil?
          fnc << 'd' if turbo
          vecs, _b, vals, _info = Numo::TinyLinalg::Lapack.send(fnc.to_sym, a.dup, b.dup, jobz: jobz)
        else
          fnc << 'x'
          il = vals_range.first(1)[0] + 1
          iu = vals_range.last(1)[0] + 1
          _a, _b, _m, vals, vecs, _ifail, _info = Numo::TinyLinalg::Lapack.send(
            fnc.to_sym, a.dup, b.dup, jobz: jobz, range: 'I', il: il, iu: iu
          )
        end
      else
        fnc = %w[d s].include?(bchr) ? "#{bchr}syev" : "#{bchr}heev"
        if vals_range.nil?
          fnc << 'd' if turbo
          vecs, vals, _info = Numo::TinyLinalg::Lapack.send(fnc.to_sym, a.dup, jobz: jobz)
        else
          fnc << 'r'
          il = vals_range.first(1)[0] + 1
          iu = vals_range.last(1)[0] + 1
          _a, _m, vals, vecs, _isuppz, _info = Numo::TinyLinalg::Lapack.send(
            fnc.to_sym, a.dup, jobz: jobz, range: 'I', il: il, iu: iu
          )
        end
      end

      vecs = nil if vals_only

      [vals, vecs]
    end

    # Computes the matrix or vector norm.
    #
    #   |  ord  |  matrix norm           | vector norm                 |
    #   | ----- | ---------------------- | --------------------------- |
    #   |  nil  | Frobenius norm         | 2-norm                      |
    #   | 'fro' | Frobenius norm         |  -                          |
    #   | 'nuc' | nuclear norm           |  -                          |
    #   | 'inf' | x.abs.sum(axis:-1).max | x.abs.max                   |
    #   |    0  |  -                     | (x.ne 0).sum                |
    #   |    1  | x.abs.sum(axis:-2).max | same as below               |
    #   |    2  | 2-norm (max sing_vals) | same as below               |
    #   | other |  -                     | (x.abs**ord).sum**(1.0/ord) |
    #
    # @example
    #   require 'numo/tiny_linalg'
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   # matrix norm
    #   x = Numo::DFloat[[1, 2, -3, 1], [-4, 1, 8, 2]]
    #   pp Numo::Linalg.norm(x)
    #   # => 10
    #
    #   # vector norm
    #   x = Numo::DFloat[3, -4]
    #   pp Numo::Linalg.norm(x)
    #   # => 5
    #
    # @param a [Numo::NArray] The matrix or vector (>= 1-dimensinal NArray)
    # @param ord [String/Numeric] The order of the norm.
    # @param axis [Integer/Array] The applied axes.
    # @param keepdims [Bool] The flag indicating whether to leave the normed axes in the result as dimensions with size one.
    # @return [Numo::NArray/Numeric] The norm of the matrix or vectors.
    def norm(a, ord = nil, axis: nil, keepdims: false) # rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/MethodLength, Metrics/PerceivedComplexity
      a = Numo::NArray.asarray(a) unless a.is_a?(Numo::NArray)

      return 0.0 if a.empty?

      # for compatibility with Numo::Linalg.norm
      if ord.is_a?(String)
        if ord == 'inf'
          ord = Float::INFINITY
        elsif ord == '-inf'
          ord = -Float::INFINITY
        end
      end

      if axis.nil?
        norm = case a.ndim
               when 1
                 Numo::TinyLinalg::Blas.send(:"#{blas_char(a)}nrm2", a) if ord.nil? || ord == 2
               when 2
                 if ord.nil? || ord == 'fro'
                   Numo::TinyLinalg::Lapack.send(:"#{blas_char(a)}lange", a, norm: 'F')
                 elsif ord.is_a?(Numeric)
                   if ord == 1
                     Numo::TinyLinalg::Lapack.send(:"#{blas_char(a)}lange", a, norm: '1')
                   elsif !ord.infinite?.nil? && ord.infinite?.positive?
                     Numo::TinyLinalg::Lapack.send(:"#{blas_char(a)}lange", a, norm: 'I')
                   end
                 end
               else
                 if ord.nil?
                   b = a.flatten.dup
                   Numo::TinyLinalg::Blas.send(:"#{blas_char(b)}nrm2", b)
                 end
               end
        unless norm.nil?
          norm = Numo::NArray.asarray(norm).reshape(*([1] * a.ndim)) if keepdims
          return norm
        end
      end

      if axis.nil?
        axis = Array.new(a.ndim) { |d| d }
      else
        case axis
        when Integer
          axis = [axis]
        when Array, Numo::NArray
          axis = axis.flatten.to_a
        else
          raise ArgumentError, "invalid axis: #{axis}"
        end
      end

      raise ArgumentError, "the number of dimensions of axis is inappropriate for the norm: #{axis.size}" unless axis.size == 1 || axis.size == 2
      raise ArgumentError, "axis is out of range: #{axis}" unless axis.all? { |ax| (-a.ndim...a.ndim).cover?(ax) }

      if axis.size == 1
        ord ||= 2
        raise ArgumentError, "invalid ord: #{ord}" unless ord.is_a?(Numeric)

        ord_inf = ord.infinite?
        if ord_inf.nil?
          case ord
          when 0
            a.class.cast(a.ne(0)).sum(axis: axis, keepdims: keepdims)
          when 1
            a.abs.sum(axis: axis, keepdims: keepdims)
          else
            (a.abs**ord).sum(axis: axis, keepdims: keepdims)**1.fdiv(ord)
          end
        elsif ord_inf.positive?
          a.abs.max(axis: axis, keepdims: keepdims)
        else
          a.abs.min(axis: axis, keepdims: keepdims)
        end
      else
        ord ||= 'fro'
        raise ArgumentError, "invalid ord: #{ord}" unless ord.is_a?(String) || ord.is_a?(Numeric)
        raise ArgumentError, "invalid axis: #{axis}" if axis.uniq.size == 1

        r_axis, c_axis = axis.map { |ax| ax.negative? ? ax + a.ndim : ax }

        norm = if ord.is_a?(String)
                 raise ArgumentError, "invalid ord: #{ord}" unless %w[fro nuc].include?(ord)

                 if ord == 'fro'
                   Numo::NMath.sqrt((a.abs**2).sum(axis: axis))
                 else
                   b = a.transpose(c_axis, r_axis).dup
                   gesvd = :"#{blas_char(b)}gesvd"
                   s, = Numo::TinyLinalg::Lapack.send(gesvd, b, jobu: 'N', jobvt: 'N')
                   s.sum(axis: -1)
                 end
               else
                 ord_inf = ord.infinite?
                 if ord_inf.nil?
                   case ord
                   when -2
                     b = a.transpose(c_axis, r_axis).dup
                     gesvd = :"#{blas_char(b)}gesvd"
                     s, = Numo::TinyLinalg::Lapack.send(gesvd, b, jobu: 'N', jobvt: 'N')
                     s.min(axis: -1)
                   when -1
                     c_axis -= 1 if c_axis > r_axis
                     a.abs.sum(axis: r_axis).min(axis: c_axis)
                   when 1
                     c_axis -= 1 if c_axis > r_axis
                     a.abs.sum(axis: r_axis).max(axis: c_axis)
                   when 2
                     b = a.transpose(c_axis, r_axis).dup
                     gesvd = :"#{blas_char(b)}gesvd"
                     s, = Numo::TinyLinalg::Lapack.send(gesvd, b, jobu: 'N', jobvt: 'N')
                     s.max(axis: -1)
                   else
                     raise ArgumentError, "invalid ord: #{ord}"
                   end
                 else
                   r_axis -= 1 if r_axis > c_axis
                   if ord_inf.positive?
                     a.abs.sum(axis: c_axis).max(axis: r_axis)
                   else
                     a.abs.sum(axis: c_axis).min(axis: r_axis)
                   end
                 end
               end
        if keepdims
          norm = Numo::NArray.asarray(norm) unless norm.is_a?(Numo::NArray)
          norm = norm.reshape(*([1] * a.ndim))
        end

        norm
      end
    end

    # Computes the Cholesky decomposition of a symmetric / Hermitian positive-definite matrix.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   s = Numo::DFloat.new(3, 3).rand - 0.5
    #   a = s.transpose.dot(s)
    #   u = Numo::Linalg.cholesky(a)
    #
    #   pp u
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[0.532006, 0.338183, -0.18036],
    #   #  [0, 0.325153, 0.011721],
    #   #  [0, 0, 0.436738]]
    #
    #   pp (a - u.transpose.dot(u)).abs.max
    #   # => 1.3877787807814457e-17
    #
    #   l = Numo::Linalg.cholesky(a, uplo: 'L')
    #
    #   pp l
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[0.532006, 0, 0],
    #   #  [0.338183, 0.325153, 0],
    #   #  [-0.18036, 0.011721, 0.436738]]
    #
    #   pp (a - l.dot(l.transpose)).abs.max
    #   # => 1.3877787807814457e-17
    #
    # @param a [Numo::NArray] The n-by-n symmetric matrix.
    # @param uplo [String] Whether to compute the upper- or lower-triangular Cholesky factor ('U' or 'L').
    # @return [Numo::NArray] The upper- or lower-triangular Cholesky factor of a.
    def cholesky(a, uplo: 'U')
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}potrf"
      c, _info = Numo::TinyLinalg::Lapack.send(fnc, a.dup, uplo: uplo)

      case uplo
      when 'U'
        c.triu
      when 'L'
        c.tril
      else
        raise ArgumentError, "invalid uplo: #{uplo}"
      end
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` with the Cholesky factorization of `A`.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   s = Numo::DFloat.new(3, 3).rand - 0.5
    #   a = s.transpose.dot(s)
    #   u = Numo::Linalg.cholesky(a)
    #
    #   b = Numo::DFloat.new(3).rand
    #   x = Numo::Linalg.cho_solve(u, b)
    #
    #   puts (b - a.dot(x)).abs.max
    #   => 0.0
    #
    # @param a [Numo::NArray] The n-by-n cholesky factor.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param uplo [String] Whether to compute the upper- or lower-triangular Cholesky factor ('U' or 'L').
    # @return [Numo::NArray] The solution vector or matrix `X`.
    def cho_solve(a, b, uplo: 'U')
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise ArgumentError, "incompatible dimensions: a.shape[0] = #{a.shape[0]} != b.shape[0] = #{b.shape[0]}" if a.shape[0] != b.shape[0]

      bchr = blas_char(a, b)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      fnc = :"#{bchr}potrs"
      x, _info = Numo::TinyLinalg::Lapack.send(fnc, a, b.dup, uplo: uplo)
      x
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
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @return [Float/Complex] The determinant of `a`.
    def det(a)
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      getrf = :"#{bchr}getrf"
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
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param driver [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @return [Numo::NArray] The inverse matrix of `a`.
    def inv(a, driver: 'getrf', uplo: 'U') # rubocop:disable Lint/UnusedMethodArgument
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      getrf = :"#{bchr}getrf"
      getri = :"#{bchr}getri"

      lu, piv, info = Numo::TinyLinalg::Lapack.send(getrf, a.dup)
      if info.zero?
        Numo::TinyLinalg::Lapack.send(getri, lu, piv)[0]
      elsif info.positive?
        raise 'the factor U is singular, and the inverse matrix could not be computed.'
      else
        raise "the #{-info}-th argument of getrf had illegal value"
      end
    end

    # Computes the (Moore-Penrose) pseudo-inverse of a matrix using singular value decomposition.
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
    # @param driver [String] The LAPACK driver to be used ('svd' or 'sdd').
    # @param rcond [Float] The threshold value for small singular values of `a`, default value is `a.shape.max * EPS`.
    # @return [Numo::NArray] The pseudo-inverse of `a`.
    def pinv(a, driver: 'svd', rcond: nil)
      s, u, vh = svd(a, driver: driver, job: 'S')
      rcond = a.shape.max * s.class::EPSILON if rcond.nil?
      rank = s.gt(rcond * s[0]).count

      u = u[true, 0...rank] / s[0...rank]
      u.dot(vh[0...rank, true]).conj.transpose
    end

    # Computes the QR decomposition of a matrix.
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
    # @return [Numo::NArray] if mode='r'.
    # @return [Array<Numo::NArray>] if mode='reduce' or 'economic' or 'raw'.
    def qr(a, mode: 'reduce')
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, "invalid mode: #{mode}" unless %w[reduce r economic raw].include?(mode)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      geqrf = :"#{bchr}geqrf"
      qr, tau, = Numo::TinyLinalg::Lapack.send(geqrf, a.dup)

      return [qr, tau] if mode == 'raw'

      m, n = qr.shape
      r = m > n && %w[economic raw].include?(mode) ? qr[0...n, true].triu : qr.triu

      return r if mode == 'r'

      org_ung_qr = %w[d s].include?(bchr) ? :"#{bchr}orgqr" : :"#{bchr}ungqr"

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

    # Solves linear equation `A * x = b` or `A * X = B` for `x` from square matrix `A`.
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
    # @param a [Numo::NArray] The n-by-n square matrix.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param driver [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @param uplo [String] This argument is for compatibility with Numo::Linalg.solver, and is not used.
    # @return [Numo::NArray] The solusion vector / matrix `X`.
    def solve(a, b, driver: 'gen', uplo: 'U') # rubocop:disable Lint/UnusedMethodArgument
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a, b)
      raise ArgumentError, "invalid array type: #{a.class}, #{b.class}" if bchr == 'n'

      gesv = :"#{bchr}gesv"
      Numo::TinyLinalg::Lapack.send(gesv, a.dup, b.dup)[1]
    end

    # Solves linear equation `A * x = b` or `A * X = B` for `x` assuming `A` is a triangular matrix.
    #
    # @example
    #   require 'numo/tiny_linalg'
    #
    #   Numo::Linalg = Numo::TinyLinalg unless defined?(Numo::Linalg)
    #
    #   a = Numo::DFloat.new(3, 3).rand.triu
    #   b = Numo::DFloat.eye(3)
    #
    #   x = Numo::Linalg.solve(a, b)
    #
    #   pp x
    #   # =>
    #   # Numo::DFloat#shape=[3,3]
    #   # [[16.1932, -52.0604, 30.5283],
    #   #  [0, 8.61765, -17.9585],
    #   #  [0, 0, 6.05735]]
    #
    #   pp (b - a.dot(x)).abs.max
    #   # => 4.071100642430302e-16
    #
    # @param a [Numo::NArray] The n-by-n triangular matrix.
    # @param b [Numo::NArray] The n right-hand side vector, or n-by-nrhs right-hand side matrix.
    # @param lower [Boolean] The flag indicating whether to use the lower-triangular part of `a`.
    # @return [Numo::NArray] The solusion vector / matrix `X`.
    def solve_triangular(a, b, lower: false)
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a, b)
      raise ArgumentError, "invalid array type: #{a.class}, #{b.class}" if bchr == 'n'

      trtrs = :"#{bchr}trtrs"
      uplo = lower ? 'L' : 'U'
      x, info = Numo::TinyLinalg::Lapack.send(trtrs, a, b.dup, uplo: uplo)
      raise "wrong value is given to the #{info}-th argument of #{trtrs} used internally" if info.negative?

      x
    end

    # Computes the Singular Value Decomposition (SVD) of a matrix: `A = U * S * V^T`
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
    # @param driver [String] The LAPACK driver to be used ('svd' or 'sdd').
    # @param job [String] The job option ('A', 'S', or 'N').
    # @return [Array<Numo::NArray>] The singular values and singular vectors ([s, u, vt]).
    def svd(a, driver: 'svd', job: 'A')
      raise ArgumentError, "invalid job: #{job}" unless /^[ASN]/i.match?(job.to_s)

      bchr = blas_char(a)
      raise ArgumentError, "invalid array type: #{a.class}" if bchr == 'n'

      case driver.to_s
      when 'sdd'
        gesdd = :"#{bchr}gesdd"
        s, u, vt, info = Numo::TinyLinalg::Lapack.send(gesdd, a.dup, jobz: job)
      when 'svd'
        gesvd = :"#{bchr}gesvd"
        s, u, vt, info = Numo::TinyLinalg::Lapack.send(gesvd, a.dup, jobu: job, jobvt: job)
      else
        raise ArgumentError, "invalid driver: #{driver}"
      end

      raise "the #{info.abs}-th argument had illegal value" if info.negative?
      raise 'input array has a NAN entry' if info == -4
      raise 'svd did not converge' if info.positive?

      [s, u, vt]
    end

    # @!visibility private
    def matmul(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def matrix_power(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def svdvals(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def orth(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def null_space(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def lu(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def lu_fact(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def lu_inv(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def lu_solve(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def ldl(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def cho_fact(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def cho_inv(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def eig(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def eigvals(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def eigvalsh(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def cond(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def slogdet(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def matrix_rank(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def lstsq(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end

    # @!visibility private
    def expm(*args)
      raise NotImplementedError, "#{__method__} is not yet implemented in Numo::TinyLinalg"
    end
  end
end
