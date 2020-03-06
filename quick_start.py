# -*- coding: utf-8 -*-
import pickle
import cage_data

"""
-------- Specify path and file name --------
"""
path = './'
pickle_filename = '20190304_Greyson_Cage_001.pkl'
with open ( ''.join((path, pickle_filename)), 'rb' ) as fp:
    my_cage_data = pickle.load(fp)

"""
-------- Bin data with this function: --------
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
timeframe, binned_spikes, binned_filtered_EMG are what you need.
"""
EMG_names = my_cage_data.EMG_names

my_cage_data.pre_processing_summary()