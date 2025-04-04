# Interfaces
The `mkl_fft` package provides the following interfaces which provide drop-in replacements for equivalent functions in NumPy and SciPy.

---

## NumPy interface - `mkl_fft.interfaces.numpy_fft`

This interface is a drop-in replacement for [`numpy.fft`](https://numpy.org/devdocs/reference/routines.fft.html) module and includes the following functions:

### complex-to-complex (c2c) FFTs:

`fft(x, n=None, axis=-1, norm=None, out=out)` - 1D FFT

`fft2(x, s=None, axes=(-2, -1), norm=None, out=out)` - 2D FFT

`fftn(x, s=None, axes=None, norm=None, out=out)` - ND FFT

and similar inverse FFT (`ifft*`) functions.

### real-to-complex (r2c) and complex-to-real (c2r) FFTs:

`rfft(x, n=None, axis=-1, norm=None, out=out)` - 1D r2c FFT

`rfft2(x, s=None, axes=(-2, -1), norm=None, out=out)` - 2D r2c FFT

`rfftn(x, s=None, axes=None, norm=None, out=out)` - ND r2c FFT

and similar inverse c2r FFT (`irfft*`) functions.

### Hermitian FFTs:

`hfft(x, n=None, axis=-1, norm=None, out=out)` - 1D FFT of a signal that has Hermitian symmetry

`ihfft(x, n=None, axis=-1, norm=None, out=out)` - inverse 1D FFT of a signal that has Hermitian symmetry

---

## SciPy interface - `mkl_fft.interfaces.scipy_fft`
This interface is a drop-in replacement for [`scipy.fft`](https://scipy.github.io/devdocs/reference/fft.html) module and includes the following functions:

### complex-to-complex (c2c) FFTs:

`fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)` - 1D FFT

`fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *, plan=None)` - 2D FFT

`fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None)` - ND FFT

and similar inverse FFT (`ifft*`) functions.

### real-to-complex (r2c) and complex-to-real (c2r) FFTs:

Note that `overwrite_x` keyword argument is only supported with its default value.

`rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)` - 1D r2c FFT

`rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *, plan=None)` - 2D r2c FFT

`rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None)` - ND r2c FFT

and similar inverse c2r FFT (`irfft*`) functions.


### Hermitian FFTs:

Note that `overwrite_x` keyword argument is only supported with its default value.

`hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None)` - 1D FFT of a signal that has Hermitian symmetry

`hfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *, plan=None)` - 2D FFT of a signal that has Hermitian symmetry

`hfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None)` - ND FFT of a signal that has Hermitian symmetry

and similar inverse Hermitian  FFT (`ihfft*`) functions.

### Helper Functions:

`set_workers(workers)` - Context manager for the default number of workers used

`get_workers()` - Returns the default number of workers within the current context

### Installing `mkl_fft` as FFT backend of SciPy

`mkl_fft.interfaces.scipy_fft` support the use as a backend. The following example shows how to set `mkl_fft` as FFT backend of SciPy.

```python
import numpy, scipy, mkl, mkl_fft.interfaces.scipy_fft as mkl_backend
x = numpy.random.randn(8*7).reshape((7, 8))
mkl.verbose(1)
# True

with scipy.fft.set_backend(mkl_backend, only=True):
	result = scipy.fft.fft2(x, workers=4)  # Calls `mkl_fft` backend
# MKL_VERBOSE oneMKL 2024.0 Update 2 Patch 2 Product build 20240823 for Intel(R) 64 architecture Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) with support for INT8, BF16, FP16 (limited) instructions, and Intel(R) Advanced Matrix Extensions (Intel(R) AMX) with INT8 and BF16, Lnx 2.00GHz intel_thread
# MKL_VERBOSE FFT(drfo7:8:8x8:1:1,input_strides:{0,8,1},output_strides:{0,8,1},bScale:0.0178571,tLim:1,unaligned_output,desc:0x557affb60d40) 36.11us CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:4

expected = scipy.fft.fft2(x)  # Calls default SciPy backend
numpy.allclose(result, expected)
# True
```

The following example compares the timing of `scipy.signal.fftconvolve` using the default SciPy backend versus the `mkl_fft` backend on an Intel® Xeon® CPU.

```python
import numpy, scipy, mkl_fft.interfaces.scipy_fft as mkl_backend
import timeit
shape = (4096, 2048)
a = numpy.random.randn(*shape) + 1j*numpy.random.randn(*shape)
b = numpy.random.randn(*shape) + 1j*numpy.random.randn(*shape)

t1 = timeit.timeit(lambda: scipy.signal.fftconvolve(a, b), number=10)
print(f"Time with scipy.fft default backend: {t1:.1f} seconds")
# Time with scipy.fft default backend: 58.1 seconds

with scipy.fft.set_backend(mkl_backend, only=True):
    t2 = timeit.timeit(lambda: scipy.signal.fftconvolve(a, b), number=10)

print(f"Time with OneMKL FFT backend installed: {t2:.1f} seconds")
# Time with MKL FFT backend installed: 9.1 seconds
```
