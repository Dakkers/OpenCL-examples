import numpy as np
import scipy.fftpack as fft

N = 2048 
v = np.arange(N)

v_fft = fft.fft(v)

v_fft_altered = 2*v_fft.real + 4*v_fft.imag*1j

v_final = fft.ifft(v_fft_altered)
print v_final

