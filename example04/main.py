import numpy as np
import scipy.fftpack as fft

N = 128
v = np.arange(N) + 2*np.arange(N)*1j

v_fft = fft.fft(v)

v_fft_altered = 2 * v_fft

v_final = fft.ifft(v_fft_altered)

print v_final
