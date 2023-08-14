## [Unreleased]

## [[0.3.1](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.3.0...v0.3.1)] - 2023-08-15
- Support automatic build of OpenBLAS on Windows.
  - The author does not have a Windows PC. It will probably work.
- Add OPENBLAS_VERSION constant to TinyLinalg.

## [[0.3.0](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.2.0...v0.3.0)] - 2023-08-13
- Add cholesky and cho_solve module functions to TinyLinalg.

**Breaking change**
- Change to raise NotImplementedError when calling a method not yet implemented in Numo::TinyLinalg.

## [[0.2.0](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.1.2...v0.2.0)] - 2023-08-11
**Breaking change**
- Change LAPACK function to call when array b is not given to TinyLinalg.eigh method.

## [[0.1.2](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.1.1...v0.1.2)] - 2023-08-09
- Add dsyev, ssyev, zheev, and cheev module functions to TinyLinalg::Lapack.
- Add dsyevd, ssyevd, zheevd, and cheevd module functions to TinyLinalg::Lapack.
- Add dsyevr, ssyevr, zheevr, and cheevr module functions to TinyLinalg::Lapack.
- Fix the confirmation processs whether the array b is a square matrix or not on TinyLinalg.eigh.

## [[0.1.1](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.1.0...v0.1.1)] - 2023-08-07
- Fix method of getting start and end of eigenvalue range from vals_range arguement of TinyLinalg.eigh.

## [[0.1.0](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.0.4...v0.1.0)] - 2023-08-06
- Refactor codes and update documentations.

## [[0.0.4](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.0.3...v0.0.4)] - 2023-08-06
- Add dsygv, ssygv, zhegv, and chegv module functions to TinyLinalg::Lapack.
- Add dsygvd, ssygvd, zhegvd, and chegvd module functions to TinyLinalg::Lapack.
- Add dsygvx, ssygvx, zhegvx, and chegvx module functions to TinyLinalg::Lapack.
- Add eigh module function to TinyLinalg.

## [[0.0.3](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.0.2...v0.0.3)] - 2023-08-02
- Add dgeqrf, sgeqrf, zgeqrf, and cgeqrf module functions to TinyLinalg::Lapack.
- Add dorgqr, sorgqr, zungqr, and cungqr module functions to TinyLinalg::Lapack.
- Add det module function to TinyLinalg.
- Add pinv module function to TinyLinalg.
- Add qr module function to TinyLinalg.

## [[0.0.2](https://github.com/yoshoku/numo-tiny_linalg/compare/v0.0.1...v0.0.2)] - 2023-07-26
- Add automatic build of OpenBLAS if it is not found.
- Add dgesv, sgesv, zgesv, and cgesv module functions to TinyLinalg::Lapack.
- Add dgetrf, sgetrf, zgetrf, and cgetrf module functions to TinyLinalg::Lapack.
- Add dgetri, sgetri, zgetri, and cgetri module functions to TinyLinalg::Lapack.
- Add solve module function to TinyLinalg.
- Add inv module function to TinyLinalg.

## [0.0.1] - 2023-07-14
- Initial release.
