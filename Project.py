# %%

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

# %%
