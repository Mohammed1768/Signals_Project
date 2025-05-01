import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import sounddevice as sd


N = 3 * 44100
f = np.linspace(0, 512, int(N/2))
X = fft(x)                                
x_f = 2/N * np.abs(X[:int(N/2)])
fn1, fn2 = np.random.randint(0, 512, 2)

# Noise Generation 
n = np.sin(2 * np.pi * fn1 * t) + np.sin(2 * np.pi * fn2 * t)
xn = x + n


# Plot of Noise Signal 

plt.figure()
plt.plot(t, xn)
plt.title(f"Noisy Signal (Time Domain) — bins {fn1}, {fn2}")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

#FFT Computation 
Xn = fft(xn)
mag = 2/N * np.abs(Xn[:N//2])
freq_axis = np.linspace(0, 512, N//2)

# Peak identification
peaks = np.argsort(mag)[-2:]
detected = np.sort(peaks)  # sort for readability
print(f"Detected noise bins (spectrum): {detected}")

#Noise Removal by subtracting sususoids 
x_filtered = xn \
    - np.sin(2 * np.pi * detected[0] * t) \
    - np.sin(2 * np.pi * detected[1] * t)

#Plot of signal after noise removal
plt.figure()
plt.plot(t, x_filtered)
plt.title("Filtered Signal (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# Play the filtered signal
sd.play(x_filtered, 3 * 44100)
sd.wait()
