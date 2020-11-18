# -*- coding: utf-8 -*-
from cage_data import cage_data
# ---------- Specify the path and files ---------- #
data_path = 'F:/test/'
# -------- The first file is a MATLAB version of .nev file ---------- #
nev_file = '20200320_Pop_Cage_001.mat'
# ---------- The second file is INTAN .rhd file for EMG recordings ---------- #
rhd_file = '20200320_Pop_cage__200320_133632.rhd'

# --------- Create a cage_data instance ---------- #
my_cage_data = cage_data()
# ---------- Start compiling ---------- #
empty_channels = [3, 9, 87] # We used these two channels as software reference during most experiments before 2019-07-01
my_cage_data.create(data_path, nev_file, rhd_file, empty_channels = empty_channels, do_notch = 1, has_analog = 1)
# ---------- Simply removed large amplitude artefacts. ---------- #
# ---------- Waveforms with a maximum larger than K1 times of the threshold, or the first sample larger than K2 times of
# ---------- the threshold will be removed. ---------- #
my_cage_data.clean_cortical_data(K1 = 8, K2 = 7)
# ---------- Filter EMG with LPF 10 Hz ---------- #
my_cage_data.EMG_filtering(10)
# --------- Summary ---------- #
my_cage_data.pre_processing_summary()
# ---------- Save ---------- #
# save_path = data_path
# my_cage_data.save_to_pickle(data_path, nev_file[:-4])
# del(my_cage_data)    

