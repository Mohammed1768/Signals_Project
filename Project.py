import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft
import sounddevice as sd


"""

Generate a random signal with random frequencies from the piano (40 samples)
With duration of 10 seconds

"""

N = 40 # Number of sound samples
duration = 3 # Duration of the signal

frequencies_3rd = np.array([220.00, 246.94, 130.81, 146.83, 164.81, 174.61, 196.00, 220.00])
frequencies_4th = 2 * frequencies_3rd

t = np.linspace(0, duration, 3 * 44100) # 10 seconds of time
F = np.random.choice(frequencies_3rd, N) # Random frequencies from the 4th octave
f = np.random.choice(frequencies_4th, N) # Random frequencies from the 3rd octave
T = np.random.uniform(1, 10, N) # Random time from 1 to 4 seconds

harmonics = np.array( [np.sin(2*np.pi * f[i] * t) + 
                    np.sin(2*np.pi * F[i] * t) * (t <= T[i]) for i in range(N)] )
# Create the harmonics
# Each harmonic is a sum of two sine waves: one from the 3rd octave and one from the 4th octave

x = np.sum(harmonics, axis=0) # Sum of the harmonics
# axis is set to 0 to sum along the columns

plt.plot(t, x) # Plot the sound
plt.title('Sound Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show() # Show the plot

sd.play(x, 3 * 44100) # Play the sound
sd.wait() # Wait until the sound is finished playing


"""

Add noise to the signal, plot the noisy signal

"""


N = 3 * 44100
f = np.linspace(0, 512, int(N/2))
X = fft(x)
x_f = 2/N * np.abs(X[:int(N/2)])

plt.figure()
plt.plot(f, x_f)
plt.title("Frequency Domain (Initial Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()


# choose two random frequencies for noise generation
# from the range of 0 to 512 Hz
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


# Frequency domain of the noisy signal
Xn = fft(xn)  # Compute the FFT of the noisy signal
x_f_noisy = 2 / N * np.abs(Xn[:int(N / 2)])  # Magnitude spectrum
freq_axis = np.linspace(0, 512, int(N / 2))  # Frequency axis up to 512 Hz

# Plot the frequency domain of the noisy signal
plt.figure()
plt.plot(freq_axis, x_f_noisy)
plt.title("Frequency Domain (Noisy Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()


"""
Remove noise from the signal using FFT
Plot the filtered signal

"""



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

# Frequency domain of the filtered signal
X_filtered = fft(x_filtered)  # Compute the FFT of the filtered signal
x_f_filtered = 2 / N * np.abs(X_filtered[:int(N / 2)])  # Magnitude spectrum
freq_axis = np.linspace(0, 512, int(N / 2))  # Frequency axis up to 512 Hz

# Plot the frequency domain of the filtered signal
plt.figure()
plt.plot(freq_axis, x_f_filtered)
plt.title("Frequency Domain (Filtered Signal)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()
plt.show()

# Play the filtered signal
sd.play(x_filtered, 3 * 44100)
sd.wait()
