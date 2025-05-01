# %%

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


frequencies_3rd = np.array([220.00, 246.94, 130.81, 146.83, 164.81, 174.61, 196.00, 220.00])
frequencies_4th = 2 * frequencies_3rd

N = 5 # Number of sound samples

t = np.linspace(0, 3, 3 * 44100) # 3 seconds of time
F = np.random.choice(frequencies_3rd, N) # Random frequencies from the 4th octave
f = np.random.choice(frequencies_4th, N) # Random frequencies from the 3rd octave
T = np.random.choice(1, 4, N) # Random time from 1 to 4 seconds

harmonics = np.sin(2*np.pi*f*t) + np.sin(2*np.pi*F*t) * (t<=T) # Harmonics of the frequencies
x = np.sum(harmonics, axis=0) # Sum of the harmonics

plt.plot(t, x) # Plot the sound
plt.title('Sound Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show() # Show the plot

# %%
