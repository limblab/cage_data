import pickle

"""
-------- Specify path and file name --------
"""
path = 'F:/test/'
pickle_filename = '20200320_Pop_Cage_001.pkl'
with open ( ''.join((path, pickle_filename)), 'rb' ) as fp:
    my_cage_data = pickle.load(fp)
my_cage_data.pre_processing_summary()
"""
-------- Here are important attributes of the cage data structure. --------
"""
#-------- spike times --------#
spikes = my_cage_data.spikes
#-------- spike channel id --------#
spike_ch_lbl = my_cage_data.ch_lbl
#-------- spike waveforms --------#
waveforms = my_cage_data.waveforms

#-------- Muscle names --------#
EMG_names = my_cage_data.EMG_names
#-------- Raw EMG data without filtering --------#
raw_EMG = my_cage_data.EMG_diff
#-------- Filtered EMG without downsampling --------#
filtered_EMG = my_cage_data.filtered_EMG

"""
-------- Use this function to bin data: --------
"""
bin_size = 0.01
my_cage_data.bin_data(bin_size, mode = 'center')
# Here the 'mode' parameter spcifies the way of binning.
# The default setting is 'center', meaning each bin is centered on the sampling time point
# If the mode is set to 'left', then the spike counts are calculated on the left side of the sampling time point
"""
-------- Binned data can be visited like this --------
"""
timeframe = my_cage_data.binned['timeframe']
binned_spikes = my_cage_data.binned['spikes']
binned_filtered_EMG = my_cage_data.binned['filtered_EMG']

"""
-------- If smoothing is needed, use this function --------
"""
my_cage_data.smooth_binned_spikes('gaussian', 0.05)
# -------- Smoothed spikes can be obtained like this: --------#
smoothed_binned_spikes = my_cage_data.binned['spikes']
my_cage_data.pre_processing_summary()
