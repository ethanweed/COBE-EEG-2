import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotFFT (data, chans, low_cutoff, high_cutoff):
	# get time-stamp data
	t = data[:,0]

	for s in chans:
		y = data[:,s]
		Y    = np.fft.fft(y)
		freq = np.fft.fftfreq(len(y), t[1] - t[0])

		# remove DC (non-varying) component
		freq = freq[1:]
		Y = np.abs(Y[1:])

		pylab.figure()
		pylab.plot( freq, Y )
		plt.xlabel('Channel '+str(s))

		#pp.savefig()
		#yheight = max(freq[low_cutoff:high_cutoff])

		plt.xlim(low_cutoff,high_cutoff)
		#plt.ylim(0,yheight)
		
		
		
# returns a dataframe with the values from an fft, indexed by frequency values
def getPower (data, chan, low_cutoff, high_cutoff):
    # get time-stamp data
    t = data[:,0]

    for s in chan:
        y = data[:,s]
        Y    = np.fft.fft(y)
        freq = np.fft.fftfreq(len(y), t[1] - t[0])

        # remove DC (non-varying) component
        freq = freq[1:]
        Y = np.abs(Y[1:])
        #freq = freq.astype(int)
        #df = pd.DataFrame({'freq':freq, 'Y':Y})
        df = pd.DataFrame(data = Y,index = freq)
        df.index.name = 'freq'
        df.columns = ['power']
        return(df)

