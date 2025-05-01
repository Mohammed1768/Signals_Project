import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


frequencies_3rd = np.array([220.00, 246.94, 130.81, 146.83, 164.81, 174.61, 196.00, 220.00])
frequencies_4th = 2 * frequencies_3rd

N = 50 # Number of sound samples
time = 10

t = np.linspace(0, 10, 3 * 44100) # 3 seconds of time
F = np.random.choice(frequencies_3rd, N) # Random frequencies from the 4th octave
f = np.random.choice(frequencies_4th, N) # Random frequencies from the 3rd octave
T = np.random.uniform(1, 4, N) # Random time from 1 to 4 seconds

harmonics = np.array([np.sin(2 * np.pi * f[i] * t) + 
                    np.sin(2 * np.pi * F[i] * t) * (t <= T[i]) for i in range(N)])
# Create the harmonics
# Each harmonic is a sum of two sine waves: one from the 3rd octave and one from the 4th octave

x = np.sum(harmonics, axis=0) # Sum of the harmonics

plt.plot(t, x) # Plot the sound
plt.title('Sound Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show() # Show the plot
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
