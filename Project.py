
# 1 2 3 4 5 6 7 8 9 
# بسم الله الرحمن الرحيم


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import sounddevice as sd


fs = 44100  # number of samples per second
note_duration = 0.5

# Define frequencies
notes_3rd = {'C': 130.81, 'D': 146.83, 'E': 164.81, 'F': 174.61,
             'G': 196.00, 'A': 220.00, ' ': 0.0}
notes_4th = {k: 2 * v for k, v in notes_3rd.items()}
 
# Melody with rest between two phrases
melody = ['C', 'C', 'G', 'G', 'A', 'A', 'G',
          ' ',  # ← 0.5s gap in the song
          'F', 'F', 'E', 'E', 'D', 'D', 'C', ' ']

# f = third octave, F = fourth octave
f = np.array([notes_3rd[n] for n in melody])
F = np.array([notes_4th[n] for n in melody])

# Set timings
t = np.array([i * note_duration for i in range(len(melody))])
T = t + note_duration

# Time array
duration = T[-1]
time = np.linspace(0, duration, int(fs * duration))

# Initialize the signal
x = np.zeros_like(time)

# i know i shouldnt use a for loop in numpy bas ana wallahi me4 3aref a3melha ezay
for i in range(len(melody)):
    if f[i] == 0.0:
        continue  # do noting for the duration, aka el signal hateb2a fadya fel duration
    
    # Calculate start and end indices for the note
    start_idx = int(t[i] * fs)
    end_idx = int(T[i] * fs)

    note_time = np.linspace(0, note_duration, end_idx - start_idx)

    # Generate a temporary signal (two octaves)
    signal = np.sin(2*np.pi*f[i]*note_time) + np.sin(2*np.pi*F[i]*note_time)

    # add the temporary signal to the substring of x
    x[start_idx:end_idx] += signal


"""

Yehia, continue your code here
Generate noise by adding two sinusoids with random frequencies

"""


N = len(x)
f = np.linspace(0, 44100/2, int(N / 2)) # frequency axis

X = fft(x) # Compute the fourier transform of the original signal
x_f = 2/N * np.abs(X[:int(N//2)]) 

# choose two random frequencies for noise generation
# from the range of 0 to 512 Hz
fn1, fn2 = np.random.randint(0, 512, 2)

# Noise Generation 
noise = np.sin(2 * np.pi * fn1 * time) + np.sin(2 * np.pi * fn2 * time)  # Generate a noise signal
xn = x + noise  # Add noise to the original signal


# Frequency domain of the noisy signal
Xn = fft(xn)  # Compute the fourier transform of the noisy signal
Xn = (2/N) * np.abs(Xn[:int(N / 2)])  # Magnitude spectrum

"""
Yehia, continue your code here
Remove noise from the signal using the FFT

"""

# Peak identification (find indices of the two largest peaks)
peak_indices = np.argsort(Xn)[-2:]  # Get indices of the 2 largest peaks
detected_frequencies = f[peak_indices]  # Get the detected frequencies

noise = np.zeros_like(xn)  # Initialize noise array
noise += np.sin(2*np.pi*detected_frequencies[0]*time) + np.sin(2*np.pi*detected_frequencies[1]*time)  # Generate noise signal
x_filtered = xn - noise  # Remove noise from the original signal


# Frequency domain of the filtered signal
x_f_filtered = fft(x_filtered) # Compute the fourier transform of the filtered signal
x_f_filtered = 2/N * np.abs(x_f_filtered[:int(N//2)]) 


# print the noise frequencies and the detected frequencies 
# used for debugging
# print(f"Noise frequencies: {fn1}, {fn2}")
# print(f"Detected frequencies: {detected_frequencies[0]}, {detected_frequencies[1]}")


"""

Yehia, plot the signals in time and frequency domain here

"""

plt.subplot(3, 1, 1)
plt.plot(time[:7000], x[:7000])
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(time[:7000], xn[:7000])
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(time[:7000], x_filtered[:7000])
plt.grid()
plt.show()

plt.subplot(3, 1, 1)
plt.plot(f[:7000], x_f[:7000])
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(f[:7000], Xn[:7000])
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(f[:7000], x_f_filtered[:7000])
plt.grid()
plt.show()


# Play the original signal
sd.play(x, fs)
sd.wait()

# Play the noisy signal
sd.play(xn, fs)
sd.wait()

# Play the filtered signal
sd.play(x_filtered, fs)
sd.wait()