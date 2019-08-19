# audio_preprocessing.py

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from spiral import *
from scipy.ndimage import gaussian_filter

def plot(signal):
	plt.figure(1)
	plt.title('Signal Wave...')
	plt.plot(signal)
	plt.show()

args = sys.argv

if len(args) != 5:
	print("usage: python audio_preprocessing.py INFILENAME OUTFILENAME BRIGHTNESS NOISE")

INFILENAME = args[1]
OUTFILENAME = args[2]
BRIGHTNESS = int(args[3])
NOISE = int(args[4])

soundfile = wave.open(INFILENAME,'r')

#Extract Raw Audio from Wav File
signal = soundfile.readframes(-1)
signal = np.fromstring(signal, 'Int16')
plot(signal)

# scale to 0-1 range
#signal = np.interp(signal, (signal.min(), signal.max()), (0, +1))



#signal = np.abs(signal)
#signal = signal / np.max(np.abs(signal),axis=0)

# downsample to size of ppm
w, h = 500, 500
dimension = w*h

# trim end of array for even resize
signal_length = signal.shape[0]
resize_val = signal_length - (signal_length // dimension) * dimension
signal = signal[:-resize_val].copy()
print(signal.shape)

# downsample
group = (signal_length // dimension)
signal = signal.reshape(-1, group).mean(axis=1)
print(signal.shape)

# scale to rgb space
#signal = (signal * BRIGHTNESS)
signal = np.interp(signal, (signal.min(), signal.max()), (0, +255))
plot(signal)

# spiral array
 
signal = signal.reshape(-1, 500)
signal = spiral(signal.tolist())

plot(signal)
# apply gaussian smoothing
signal = np.array(signal)
signal = signal.reshape(-1, 500)
signal = gaussian_filter(signal, sigma=[NOISE, NOISE])
signal = signal.flatten()

plot(signal)

signal = np.interp(signal, (signal.min(), signal.max()), (0, +255))
plot(signal)

signal = signal.astype(int)

plot(signal)

# write to ppm

#open file
f = open(OUTFILENAME, "w")

#write header
f.write('P2 \n')
f.write('# ppg \n')
f.write('500 500 \n')
f.write('255 \n')

rgbval = 0

for val in signal:
	if val > 255:
		rgbval = 255
	else:
		rgbval = val
	f.write(str(rgbval) + ' ') #r
	#f.write(str(rgbval) + ' ') #g
	#f.write(str(rgbval) + ' ') #b
	if val%100 == 0:
		f.write('\n')



