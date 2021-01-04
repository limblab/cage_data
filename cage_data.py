import numpy as np
import _pickle as pickle
import time
import h5py
import getpass
import joblib
import os
import copy
import xlrd
from load_intan_rhd_format import read_data
from scipy import stats, signal
from intanutil.notch_filter import notch_filter
from brpylib import NevFile, NsxFile
from scipy.signal import find_peaks
from collections import defaultdict

memo1 = """Since Blackrock Python codes (brpy) are too slow when reading .nev files, we use MATLAB version of .nev files instead."""
memo2 = """Please make sure MATLAB version of .nev files are in your target directory. """

Pop_EMG_names_single = ['APB_1', 'Lum_1', 'PT_1', '1DI_1',
                        'FDP2_1', 'FCR1_1', 'FCU1_1', 'FCUR_1',
                        'FCUR_2', 'FCU1_2',	'FCR1_2', 'FDP2_2', 
                        '1DI_2', 'PT_2', 'Lum_2', 'APB_2',
                        'FPB_1', '3DI_1', 'SUP_1', 'ECU_1', 
                        'ECR_1', 'EDC1_1',	'BI_1', 'TRI_1', 
                        'TRI_2', 'BI_2', 'EDC1_2', 'ECR_2',
                        'ECU_2', 'SUP_2', '3DI_2', 'FPB_2']
"""
For the datasets collected on Pop between 2020-03 and 2020-09 using the DSPW system, channels 7 and 16 are noisy and should be taken out.
For the datasets collected on Pop after 2020-09 using the DSPW system, channels 7, 16, 3, and 12 are noisy and should be taken out.
For the datasets collected on all monkeys after 2018-12, channels 24, 25, and 26 should be taken out due to the short circuit of the adapter board.
"""

"""
In summary, for the data collected between 2020-09 and 2020-10 on Pop, the indices and names for the bad EMG channels are as below:
indices: [3, 7, 12, 16, 24, 25, 26]
names: ['1DI_1', 'FCUR_1', '1DI_2', 'FPB_1', 'TRI_2', 'BI_2', 'EDC1_2']
"""

def get_paired_EMG_index(EMG_names_single):
    EMG_names = []
    EMG_index1 = []
    EMG_index2 = []
    for i in range(len(EMG_names_single)):
        temp_str = EMG_names_single[i][:-2]
        if temp_str in EMG_names:
            continue
        else:
            for j in range(i+1, len(EMG_names_single)):
                temp_str2 = EMG_names_single[j]
                if temp_str2.find(temp_str) != -1:
                    if (temp_str2[:-2] in EMG_names) == True:
                        EMG_names.append(''.join( (temp_str2[:-2], '-3') ))
                    else:
                        EMG_names.append(temp_str)
                    EMG_index1.append(EMG_names_single.index(EMG_names_single[i]))
                    EMG_index2.append(EMG_names_single.index(EMG_names_single[j]))
    return EMG_names, EMG_index1, EMG_index2

def find_bad_EMG_index_from_list(EMG_names_single, bad_EMG):
    bad_index = []
    paired_index = []
    for each in bad_EMG:
        temp = list(each)
        if each[-1] == '1':
           temp[-1] = '2'
        elif each[-1] == '2':
            temp[-1] = '1'
        elif each[-1] == '3':
            temp[-1] = '1'
        paired_name = ''.join(temp)
        # -------- Make sure the paired EMG channel can be found ---------- #
        if paired_name in EMG_names_single:
            bad_index.append(EMG_names_single.index(each))
            paired_index.append(EMG_names_single.index(paired_name))
    return bad_index, paired_index

def delete_paired_bad_channel(EMG_names_single, bad_EMG):
    """
    If both of the two single end channels are noise, then we need to get rid of both
    This function will find out the indices of them. Deleting will be done outside of this function
    """
    def list_duplicates(seq):
        tally = defaultdict(list)
        for i,item in enumerate(seq):
            tally[item].append(i)
        return ((key,locs) for key,locs in tally.items() if len(locs)>1)

    temp = []
    for each in bad_EMG:
        temp.append(each[:-1])
    bad_paired_channel = []
    names = []
    for dup in sorted(list_duplicates(temp)):
        print( 'The paired channels of %s will be deleted'%(dup[0]) )
        name1, name2 = bad_EMG[dup[1][0]], bad_EMG[dup[1][1]]
        names.append(name1)
        names.append(name2)
        bad_paired_channel.append(EMG_names_single.index(name1))
        bad_paired_channel.append(EMG_names_single.index(name2))
    bad_EMG_post = copy.deepcopy(bad_EMG)
    for each in names:
        bad_EMG_post.remove(each)
    print('The numbers of these bad channels are %s' % (bad_paired_channel))
    return bad_paired_channel, bad_EMG_post
    
class cage_data:
    def __init__(self):
        self.meta = dict()
        self.meta['Processes by'] = getpass.getuser()
        self.meta['Processes at'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('An empty cage_data object has been created.')
        print(memo1)
        print(memo2)
        
    def create(self, path, nev_mat_file, rhd_file,      # Required data files. Set 'rhd_file' as ' ' if no wireless EMG recordings
               is_sorted = 0, empty_channels = [],      # Is the nev file sorted? Any empty channel?
               bad_EMG = [], do_notch = 1,              # Whether apply notch filter? Should remove bad EMG channels?
               has_analog = 0,                          # Analog files are for sync pulses of videos.
               comb_filter = 0):                        # Comb filter for EMG preprocessing                                                    
        """
        'nev_mat_file' is the neural data file,
        'rhd_file' is the EMG data file
        """
        if rhd_file[-3:] == 'rhd':
            self.has_EMG = 1
        else:
            self.has_EMG = 0
        if is_sorted == 0:
            self.is_sorted = 0
        else:
            self.is_sorted = 1
        if path[-1] == '/':
            self.nev_mat_file = ''.join((path, nev_mat_file))
            if self.has_EMG == 1:
                self.rhd_file = ''.join((path, rhd_file))
            else:
                self.rhd_file = []
        else:
            self.nev_mat_file = ''.join((path, '/', nev_mat_file))
            if self.has_EMG == 1:
                self.rhd_file = ''.join((path, '/', rhd_file))
            else:
                self.rhd_file = []

        if self.has_EMG == 1:
            try:
                self.date_num = int(rhd_file[:8])
            except ValueError:
                self.date_num = 0
                print('Check the file name of the .rhd file!')
            else:
                pass

            self.EMG_names, self.EMG_diff, self.EMG_timeframe = self.parse_rhd_file(self.rhd_file, 
                                                                                    do_notch, 
                                                                                    bad_EMG,
                                                                                    comb_filter)
            self.file_length = self.EMG_timeframe[-1]
            print(self.EMG_names)
            
        self.parse_nev_mat_file(self.nev_mat_file, is_sorted, empty_channels, has_analog)
        
        self.is_cortical_cleaned = False
        self.is_EMG_filtered = False
        self.is_data_binned = False
        self.is_spike_smoothed = False
        self.binned = {}
        self.pre_processing_summary()
        self.raw_data = self.parse_ns6_file()
        if self.raw_data != 0:
            print('Raw data (fs = 30kHz) was recorded along with spike data in this session!')
        else:
            print('No raw data along with this recording session.')
        
    def pre_processing_summary(self):
        if hasattr(self, 'is_sorted'):
            if self.is_sorted == 1:
                print('This is a sorted file')
            else:
                print('This is a non-sorted file')
        if hasattr(self, 'EMG_diff'):
            print('EMG filtered? -- %s' %(self.is_EMG_filtered))
        else:
            print('EMG filtered? -- %s' %('There is no EMG from DSPW system.'))
        if hasattr(self, 'EMG_names'):
            print('EMG filtered? -- %s' %(self.is_EMG_filtered))
        print('Cortical data cleaned? -- %s' %(self.is_cortical_cleaned))
        if hasattr(self, 'is_data_binned'):
            print('Data binned? -- %s' %(self.is_data_binned))
        if hasattr(self, 'is_spike_smoothed'):
            print('Spikes smoothed? -- %s' %(self.is_spike_smoothed))
    
    def parse_nev_mat_file(self, filename, is_sorted, empty_channels, has_analog):
        """
        Parse MATLAB version of .nev files
        """
        s = time.clock()
        nev_file = h5py.File(filename, 'r')['NEV']
        ch_lbl = list(np.asarray(nev_file['elec_labels']).T)
        for i in range(len(ch_lbl)):
            ch_lbl[i] = ''.join(chr(x) for x in ch_lbl[i]).strip(b'\x00'.decode())
        elec_id = list(nev_file['elec_id'][0])
        electrode_meta = dict()
        electrode_meta['elec_pin'] = list(np.asarray(nev_file['elec_pin'][0]).T)
        electrode_meta['elec_bank'] = list(np.asarray(nev_file['elec_bank']).T)
        for i in range(len(electrode_meta['elec_bank'])):
            electrode_meta['elec_bank'][i] = chr(electrode_meta['elec_bank'][i])
        thresholds = list(np.asarray(nev_file['elec_threshold'][0]).T)
        # ----------- Delete empty electrode channels, they are often used as reference ---------- #
        empty_str = []
        for each in empty_channels:
            empty_str.append(''.join(('elec', str(each))))
        bad_num = []
        for each in empty_str:
            bad_num.append(ch_lbl.index(each))
        for idx in sorted(bad_num, reverse=True):
            del(ch_lbl[idx])
            del(elec_id[idx])
            del(electrode_meta['elec_pin'][idx])
            del(electrode_meta['elec_bank'][idx])
            del(thresholds[idx])
        # ---------- Deal with actual spike data ---------- #
        time_stamp = np.asarray(nev_file['data']['spikes']['TimeStamp'])
        electrode = np.asarray(nev_file['data']['spikes']['Electrode'])
        unit = np.asarray(nev_file['data']['spikes']['Unit'])
        waveform = np.asarray(nev_file['data']['spikes']['Waveform'])
        
        s_spikes, s_waveforms = [], []
        # -------- Two conditions, one for sorted files, another for unsorted files -------- #
        # -------- Default is unsorted -------- #
        if self.is_sorted == 0:
            for each in elec_id:
                # -------- Read only unsorted waveforms, avoid cases where 'unit == 255'
                idx = np.where((electrode == each) & (unit == 0))[0]
                s_spikes.append(time_stamp[idx])
                s_waveforms.append(waveform[idx,:])
        else:
            # -------- Codes for reading sorted files -------- #
            # -------- Max number of units in each channel is set as '5' --------#
            MAX = 5
            self.sorted_ch_lbl, self.sorted_elec_id, self.sorted_unit = [], [], []
            for each in elec_id:
                for u in range(1, MAX + 1):
                    idx = np.where((electrode == each) & (unit == u))[0]
                    if len(idx) > 0:
                        s_spikes.append(time_stamp[idx])
                        s_waveforms.append(waveform[idx,:])
                        self.sorted_elec_id.append(each)
                        self.sorted_ch_lbl.append(ch_lbl[elec_id.index(each)])
                        self.sorted_unit.append(u)
                    else:
                        break
        
        self.nev_fs = nev_file['fs'][0][0]
        self.nev_duration = nev_file['duration'][0][0]
        self.electrode_meta = electrode_meta
        self.thresholds = thresholds
        self.spikes = s_spikes
        self.waveforms = s_waveforms
        self.ch_lbl = ch_lbl
        self.elec_id = elec_id
        
        if has_analog == 1:
           self.analog = {}
           self.analog['analog_fs'] = nev_file['analog_fs'][0][0]
           self.analog['analog_data'] = np.asarray(nev_file['analog_data'])
           self.analog['time_frame'] = np.arange(len(self.analog['analog_data']))/self.analog['analog_fs']
           analog_lbl = list(np.asarray(nev_file['analog_labels']).T)
           for (i, each) in enumerate(analog_lbl):
               analog_lbl[i] = ''.join(chr(x) for x in each).strip(b'\x00'.decode())
           self.analog['analog_lbl'] = analog_lbl
           self.analog['analog_MaxDigiValue'] = nev_file['analog_MaxDigiValue'][0][0]
           self.analog['analog_MaxAnalogValue'] = nev_file['analog_MaxAnalogValue'][0][0]
           self.analog['analog_MinDigiValue'] = -self.analog['analog_MaxDigiValue']
           self.analog['analog_MinAnalogValue'] = -self.analog['analog_MaxAnalogValue']
           
           # Check the video sync pulses
           if 'video_sync' in self.analog['analog_lbl']:
               self.analog['video_sync'] = self.analog['analog_data'][:, self.analog['analog_lbl'].index('video_sync')]
           elif 'kinectSync' in self.analog['analog_lbl']:
               self.analog['video_sync'] = self.analog['analog_data'][:, self.analog['analog_lbl'].index('kinectSync')]
           elif 'videosync' in self.analog['analog_lbl']:
               self.analog['video_sync'] = self.analog['analog_data'][:, self.analog['analog_lbl'].index('videosync')]
           else:
               print('No sync pulses!')
           # If the rhd_file is empty, then check whether there is any EMG channel with Cerebus analog inputs
           if not self.rhd_file:
               self.check_EMG_in_cerebus_analog()   
               self.file_length = self.nev_duration
               
           # Check the FSR data
           self.check_FSR_in_cerebus_analog()
        
        e = time.clock()
        print("%.3f s for parsing the nev-mat file" %(e-s))
        
    def parse_rhd_file(self, filename, notch, bad_EMG, comb_filter):
        rhd_data = read_data(filename)
        if self.date_num < 20190701:
            self.EMG_fs = 2011.148
        else:
            self.EMG_fs = rhd_data['frequency_parameters']['amplifier_sample_rate']
        # ---------- In default case, the items in EMG_names_single are obtained from the rhd file  -------- #
        EMG_single = rhd_data['amplifier_data']
        EMG_names_single = []
        for each in rhd_data['amplifier_channels']:
            EMG_names_single.append(each['custom_channel_name'])
        # ---------- Since the labels for Pop are not right, these lines change the labels -------- #
        if (self.date_num>20200301)&(self.date_num<20201101):
            EMG_names_single = copy.deepcopy(Pop_EMG_names_single)
        # -------- If the items in bad_EMG are numbers, these lines will find out the names -------- #
        if len(bad_EMG) > 0:
            if type(bad_EMG[0]) == int:
                bad_EMG_names = [EMG_names_single[n] for n in bad_EMG]
            elif type(bad_EMG[0]) == str:
                bad_EMG_names = bad_EMG
        else:
            bad_EMG_names = []
        
        # ---------- Delete paired bad channels -------- #
        bad_paired_channel, bad_EMG_post = delete_paired_bad_channel(EMG_names_single, bad_EMG_names)
        bad_paired_channel = sorted(bad_paired_channel, reverse = True)
        for each in bad_paired_channel:
            EMG_names_single.pop(each)
        EMG_single = np.delete(EMG_single, bad_paired_channel, axis = 0)
        # ---------- To get paired EMG channels for software diffrence ---------- #
        EMG_names, EMG_index1, EMG_index2 = get_paired_EMG_index(EMG_names_single)

        EMG_diff = []
        for i in range(len(EMG_index1)):
            EMG_diff.append(EMG_single[EMG_index1[i], :] - EMG_single[EMG_index2[i], :])
        
        # ---------- Based on the list in bad_EMG, substitute some channels with single end EMG ---------- #
        if bad_EMG_post:
            bad_idx, paired_idx = find_bad_EMG_index_from_list(EMG_names_single, bad_EMG_post)
            for (i,each) in enumerate(bad_EMG_post):
                target_idx = EMG_names.index(each[:-2])
                EMG_diff[target_idx] = EMG_single[paired_idx[i], :]
                print("For noisy channel %s, use only one single end channel." %(each[:-2]))
                lost_idx = np.where(EMG_diff[target_idx]<-6300)[0]
                if lost_idx.size > 0:
                    EMG_diff[target_idx][lost_idx] = EMG_diff[target_idx][lost_idx[0]-10]
        
        # ---------- Apply artifacts rejection on EMG_diff ----------- #
        """
        For all dataset, artifacts rejection is necessary, must be done
        """
        EMG_diff = self.EMG_art_rej(EMG_diff)
        
        # ---------- Apply notch filter on EMG_diff ---------- #
        if notch == 1:
           print('Applying notch filter.')
           bnotch, anotch =  signal.iirnotch(60, 30, self.EMG_fs)
           for (i, each) in enumerate(EMG_diff): 
               EMG_diff[i] = signal.filtfilt(bnotch, anotch, each)
        else:
            print('No notch filter is applied.')
        # ---------- Apply comb filter on EMG_diff ----------- #
        """
        For dataset between 2020-06 and 2020-09, a comb filter is necessary
        """
        if comb_filter == 1:
            EMG_diff = self.apply_comb_filter(EMG_diff, self.EMG_fs)
            print('Comb filter has been applied, too.')
        
        EMG_diff = np.asarray(EMG_diff)
        # ---------- Dealing with sync ----------- #
        sync_line0 = rhd_data['board_dig_in_data'][0]
        sync_line1 = rhd_data['board_dig_in_data'][1]
        d0 = np.where(sync_line0 == True)[0]
        d1 = np.where(sync_line1 == True)[0]
        ds = int(d1[0] - int((d1[0]-d0[0])*0.2))
        de = int(d1[-1] + int((d0[-1]-d1[-1])*0.2))
        rhd_timeframe = np.arange(de-ds+1)/self.EMG_fs
        return EMG_names, list(EMG_diff[:, ds:de]), rhd_timeframe
        
    def clean_cortical_data(self, K1 = 8, K2 = 8):
        # ---------- K1 and K2 sets a threshold for high amplitude noise cancelling ----------#
        if hasattr(self, 'thresholds'):
            for i in range(len(self.waveforms)):
                bad_waveform_ind = []
                thr = abs(self.thresholds[i])
                for j in range(np.size(self.waveforms[i], 0)):
                    if max(abs( self.waveforms[i][j,:] )) > K1*thr:
                        bad_waveform_ind.append(j)
                    if abs(self.waveforms[i][j, 0]) > K2*thr:
                        bad_waveform_ind.append(j)
                self.waveforms[i] = np.delete(self.waveforms[i], bad_waveform_ind, axis = 0)
                self.spikes[i] = np.delete(self.spikes[i], bad_waveform_ind)
                self.is_cortical_cleaned = True
        else:
            print('This function may not be applied to this version of data file.')

    def EMG_filtering(self, f_Hz):
        fs = self.EMG_fs
        raw_EMG_data = self.EMG_diff
        filtered_EMG = []    
        bhigh, ahigh = signal.butter(4,50/(fs/2), 'high')
        blow, alow = signal.butter(4,f_Hz/(fs/2), 'low')
        for each in raw_EMG_data:
            temp = signal.filtfilt(bhigh, ahigh, each)
            temp = self.EMG_art_rej_single_channel(temp)
            f_abs_emg = signal.filtfilt(blow ,alow, np.abs(temp))
            filtered_EMG.append(f_abs_emg)   
        self.filtered_EMG = filtered_EMG
        print('All EMG channels have been filtered.')
        self.is_EMG_filtered = True
            
    def bin_spikes(self, bin_size, mode = 'center'):
        print('Binning spikes with %.4f s' % (bin_size))
        binned_spikes = []
        if self.has_EMG == 1:
            bin_start = self.EMG_timeframe[0]
        else:
            bin_start = 0.0
        if mode == 'center':
            bins = np.arange(bin_size - bin_size/2, 
                             self.file_length + bin_size/2, bin_size)
        elif mode == 'left':
            bins = np.arange(bin_start, self.file_length, bin_size)
        bins = bins.reshape((len(bins),))
        for each in self.spikes:
            each = each/self.nev_fs
            each = each.reshape((len(each),))
            out, _ = np.histogram(each, bins)
            binned_spikes.append(out)
        return bins[1:], binned_spikes        
      
    def EMG_downsample(self, new_fs):
        if hasattr(self, 'filtered_EMG'):
            down_sampled = []
            n = self.EMG_fs/new_fs
            length = int(np.floor(np.size(self.filtered_EMG[0])/n))
            for each in self.filtered_EMG:
                temp = []
                for i in range( 1, length ):
                    temp.append(each[int(np.floor(i*n))])
                temp = np.asarray(temp)
                down_sampled.append(temp)
            print('Filtered EMGs have been downsampled')
            return down_sampled
        else:
            print('Filter EMG first!')
            return 0
        
    def FSR_data_downsample(self, new_fs):
        if 'FSR_data' in self.analog.keys():
            down_sampled = []
            n = self.analog['analog_fs']/new_fs
            length = int(np.floor(np.size(self.analog['FSR_data'][0])/n))
            for each in self.analog['FSR_data']:
                temp = []
                for i in range( 1, length ):
                    temp.append(each[int(np.floor(i*n))])
                temp = np.asarray(temp)
                down_sampled.append(temp)
            print('FSR data have been downsampled')
            return down_sampled
        else:
            print('There is no FSR data in this dataset, please check')
            return 0
        
    def bin_data(self, bin_size, mode = 'center'):
        if not hasattr(self, 'binned'):
            self.binned = {}
        if not hasattr(self, 'has_EMG'):
            if hasattr(self, 'EMG_names'):
                self.has_EMG = 1
            else:
                self.has_EMG = 0
        self.binned['timeframe'], self.binned['spikes'] = self.bin_spikes(bin_size, mode)
        if self.has_EMG == 1:
            self.binned['filtered_EMG'] = self.EMG_downsample(1/bin_size)
            truncated_len = min(len(self.binned['filtered_EMG'][0]), len(self.binned['spikes'][0]))
            for (i, each) in enumerate(self.binned['spikes']):
                self.binned['spikes'][i] = each[:truncated_len]
            for (i, each) in enumerate(self.binned['filtered_EMG']):
                self.binned['filtered_EMG'][i] = each[:truncated_len]
            self.binned['timeframe'] = self.binned['timeframe'][:truncated_len]
        if hasattr(self, 'analog'):
            self.binned['FSR_data'] = self.FSR_data_downsample(1/bin_size)
        self.is_data_binned = True
        print('Data have been binned.')
    
    def smooth_binned_spikes(self, kernel_type, kernel_SD, sqrt = 0):
        smoothed = []
        if self.binned:
            if sqrt == 1:
               for (i, each) in enumerate(self.binned['spikes']):
                   self.binned['spikes'][i] = np.sqrt(each)
            bin_size = self.binned['timeframe'][1] - self.binned['timeframe'][0]
            kernel_hl = np.ceil( 3 * kernel_SD / bin_size )
            normalDistribution = stats.norm(0, kernel_SD)
            x = np.arange(-kernel_hl*bin_size, (kernel_hl+1)*bin_size, bin_size)
            kernel = normalDistribution.pdf(x)
            if kernel_type == 'gaussian':
                pass
            elif kernel_type == 'half_gaussian':
               for i in range(0, int(kernel_hl)):
                    kernel[i] = 0
            n_sample = np.size(self.binned['spikes'][0])
            nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + int(kernel_hl)] 
            for each in self.binned['spikes']:
                temp1 = np.convolve(kernel,each)
                temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)]/nm
                smoothed.append(temp2)
            print('The binned spikes have been smoothed.')
            self.binned['spikes'] = smoothed
            self.is_spike_smoothed = True
        else:
            print('Bin spikes first!')
            
    def save_to_pickle(self, save_path, file_name):
        if save_path[-1] == '/':
            save_name = save_path + file_name + '.pkl'
        else:
            save_name = save_path + '/' + file_name + '.pkl'
        with open (save_name, 'wb') as fp:
            pickle.dump(self, fp)
        print('Save to %s successfully' %(save_name))
        
    def ximea_video_sync(self):
        if hasattr(self, 'analog'):
           sync_pulse = self.analog['video_sync']
           M = np.max(sync_pulse)
           if (M>40000)|(M<10000):
               print('The sync pulses may be problematic, please check')
           sync_pulse[np.where(sync_pulse<M/3)[0]] = 0
           sync_pulse[np.where(sync_pulse>M/3)[0]] = 32000
           self.analog['video_sync_timeframe'] = np.arange(len(self.analog['video_sync']))/self.analog['analog_fs']
           diff_sync_pulse = np.diff(sync_pulse)
           peaks, properties = find_peaks(diff_sync_pulse,prominence=(0.5*np.max(sync_pulse), None))
           peaks = list(peaks)
           video_timestamps = self.analog['video_sync_timeframe'][peaks]
        else:
            print('No video sync signals in this file')
            peaks = 0
            video_timestamps = 0
        return video_timestamps
        
    def clean_cortical_data_with_classifier(self, clf_path, clf_file, clf_type = 'sklearn'):
        # -------- Designed only for files without spike sorting -------- #       
        # -------- The most easy to use clfs are those trained using sklearns -------- #
        # -------- Codes for clfs from pytorch will be finished later -------- #
        self.bad_waveforms = []
        waveforms = self.waveforms
        if clf_type == 'sklearn':
            clf = joblib.load(clf_path + clf_file)
            for i, each in enumerate(waveforms):
                res = clf.predict(each)
                bad_idx = np.where(res == 1)[0]
                if len(bad_idx) > 0:
                    self.bad_waveforms.append(each[bad_idx, :])
                    self.waveforms[i] = np.delete(self.waveforms[i], bad_idx, axis = 0)
                    self.spikes[i] = np.delete(self.spikes[i], bad_idx)
            
    def parse_ns6_file(self):
        ns6_file_name = self.nev_mat_file[:-4]+'.ns6'
        if os.path.exists(ns6_file_name) == False:
            print('There is no .ns6 file along with this .nev file')
            return 0
        else:
            ns6_file = NsxFile(ns6_file_name)
            _raw_data = ns6_file.getdata()
            ns6_file.close()
            raw_data = dict()
            raw_data['data'] = _raw_data['data'].T
            raw_data['fs'] = _raw_data['samp_per_s']
            raw_data['elec_id'] = _raw_data['elec_ids']
            raw_data['ch_lbl'] = list()
            if hasattr(self, 'elec_id'):
                for i, each in enumerate(raw_data['elec_id']):
                    if each in self.elec_id:
                        raw_data['ch_lbl'].append(self.ch_lbl[self.elec_id.index(each)])
                    else:
                        raw_data['ch_lbl'].append('No elec label')
            else:
                raw_data['ch_lbl'] = 0
            raw_data['timeframe'] = np.arange(len(raw_data['data']))/raw_data['fs']
            return raw_data
    
    def check_EMG_in_cerebus_analog(self):
        idx = []
        for i, lbl in enumerate(self.analog['analog_lbl']):
            if 'EMG' in lbl:
                idx.append(i)
        if idx:
            self.EMG_names, self.EMG_diff = [], []
            for i in idx:
                self.EMG_names.append(self.analog['analog_lbl'][i])
                self.EMG_diff.append(self.analog['analog_data'][:, i])
            self.EMG_timeframe = np.arange(len(self.EMG_diff[0]))/self.analog['analog_fs']
            self.EMG_fs = self.analog['analog_fs']
            print('This file contains EMG acquired by Cerebus system.')
            print(self.EMG_names)
            self.has_EMG = 1
            
    def get_EMG_idx(self, EMG_list):
        e_flag = False
        if 'EMG' in self.EMG_names[0]:
            e_flag = True
        EMG_names = np.asarray(self.EMG_names)
        
        idx = []
        for each in EMG_list:
            if (e_flag == True)&('EMG' not in each):
                each = 'EMG_' + each
            idx.append(np.where(EMG_names == each)[0])
        return np.asarray(idx).reshape((len(idx), ))
    
    def check_FSR_in_cerebus_analog(self, lp_filter = True):
        """
        A function to check the existence of the FSR data, and to read and filter them
        A lowpass filter is used to filter the FSR data, fc = 10 Hz
        """
        blow, alow = signal.butter(4, 10/(self.analog['analog_fs']/2), 'low')
        idx = []
        for i, lbl in enumerate(self.analog['analog_lbl']):
            if 'FSR' in lbl:
                idx.append(i)
        if idx:
            self.analog['FSR_data'] = []
            for i in idx:
                temp = self.analog['analog_data'][:, i]
                temp = (temp-self.analog['analog_MinDigiValue'])/32768*self.analog['analog_MaxAnalogValue']/1000
                if lp_filter == True:
                    self.analog['FSR_data'].append(signal.filtfilt(blow, alow, temp))
                else:
                    self.analog['FSR_data'].append(temp)
            print('This file contains FSR data.')
       
    def apply_comb_filter(self, input_signal, fs, f_list = [120, 180, 240, 300, 360], Q = 30):
        """
        Here input_signal is a list
        """
        output_signal = input_signal
        b, a = [], []
        for i in range(len(f_list)):
            b_temp, a_temp = signal.iirnotch(f_list[i], Q, fs)
            b.append(b_temp)
            a.append(a_temp)
        for i in range(len(input_signal)):
            for j in range(len(f_list)):
                output_signal[i] = signal.filtfilt(b[j], a[j], input_signal[i])
        return output_signal

    def EMG_art_rej(self, data_list, k = 8, L = 8):
        print('Rejecting high amplitude EMG artifacts.')
        data_list_post = []
        for data in data_list:
            c = np.where(abs(data)>k*np.std(data))[0]
            idx = []
            for each in c[:-2]:
                idx.append(list(np.arange(each-L, each+L)))
            u_idx = sorted(set(idx[0]).union(*idx))
            u_idx = np.asarray(u_idx)
            over_idx = np.where(u_idx>len(data)-1)[0]
            u_idx = list(np.delete(u_idx, over_idx))
            subs = np.random.rand(len(u_idx))*np.std(data)
            data[u_idx] = subs
            data_list_post.append(data)
        return data_list_post     
                
    def EMG_art_rej_single_channel(self, data, k = 8, L = 8):
        #print('Rejecting high amplitude EMG artifacts on single channel.')
        c = np.where(abs(data)>k*np.std(data))[0]
        idx = []
        for each in c:
            idx.append(list(np.arange(each-L, each+L)))
        u_idx = sorted(set(idx[0]).union(*idx))
        u_idx = np.asarray(u_idx)
        over_idx = np.where(u_idx>len(data)-1)[0]
        u_idx = list(np.delete(u_idx, over_idx))
        subs = np.random.rand(len(u_idx))*np.std(data)
        data[u_idx] = subs
        return data             
        
    def read_behavior_tags(self, path, file_name):
        """
        Reading in the type and the timing for each behavior segment from an xls file
        If there is an xls file with behavior information with one .nev file, this 
        function will create a dictionary to store the behavior information.
        """
        video_timeframe = self.ximea_video_sync()
        if path[-1] != '/':
            path = path+'/'
        try:
            data = xlrd.open_workbook(path+file_name)
        except IOError:
            print('Cannot open the file!')
        else:
            table = data.sheets()[0]
            start = [int(x) for x in table.col_values(0)[1:]]
            ends = [int(x) for x in table.col_values(1)[1:]]
            tags = table.col_values(3)[1:]
            self.behave_tags = dict()
            self.behave_tags['start_time'] = list(video_timeframe[start])
            self.behave_tags['end_time'] = list(video_timeframe[ends])
            self.behave_tags['tag'] = tags

    def get_elec_idx(self, elec_num):
        """
        To get the idx of electrodes specified by elec_num
        elec_num: a list containing the number of bad channels
        """
        idx = []
        for each in elec_num:
            if 'elec'+str(each) in self.ch_lbl:
                temp = self.ch_lbl.index('elec'+str(each))
                idx.append(temp)
        return idx

    def del_bad_chs(self, elec_num):
        """
        To get rid of everything about the bad channels from the data structure
        """
        idx = self.get_elec_idx(elec_num)
        for idx in sorted(idx, reverse=True):
            del(self.ch_lbl[idx])
            del(self.elec_id[idx])
            del(self.thresholds[idx])
            del(self.waveforms[idx])
            del(self.spikes[idx])
            del(self.electrode_meta['elec_pin'][idx])
            del(self.electrode_meta['elec_bank'][idx])
        
    def get_single_data_segment(self, behave_time, behave_label, requires_raw_EMG = False, requires_spike_timing = False):
        """
        This function is designed to get a data segment corresponding to the tag extracted from videos
        behave_time is a list with two elements, each of which is time in seconds
        behave_label is a string for describing the monkey's behavior
        """
        timeframe = self.binned['timeframe']
        binned_spikes = np.asarray(self.binned['spikes']).T
        emgs = np.asarray(self.binned['filtered_EMG']).T
        if 'FSR_data' in self.analog.keys():
            fsrs = np.asarray( self.binned['FSR_data'] ).T
        t = behave_time  
        idx = np.where((timeframe>t[0]) & (timeframe<t[1]))[0]
        seg_binned_spikes = binned_spikes[idx, :] 
        seg_emgs = emgs[idx, :]
        if 'FSR_data' in self.analog.keys():
            seg_fsrs = fsrs[idx, :]
        seg_timeframe = timeframe[idx] 
        seg_emg_names = self.EMG_names
        seg_unit_names = self.ch_lbl
        behave_dict = dict()
        behave_dict['spike'] = seg_binned_spikes
        behave_dict['EMG'] = seg_emgs
        behave_dict['timeframe'] = seg_timeframe
        behave_dict['label'] = behave_label
        behave_dict['EMG_names'] = seg_emg_names
        behave_dict['unit_names'] = seg_unit_names
        if requires_raw_EMG == True:
            idx_raw_EMG = np.where((self.EMG_timeframe>t[0]) & (self.EMG_timeframe<t[1]))[0]
            behave_dict['raw_EMG'] = np.asarray(self.EMG_diff).T[idx_raw_EMG, :]
            behave_dict['raw_EMG_timeframe'] = self.EMG_timeframe[idx_raw_EMG]
            behave_dict['raw_EMG_fs'] = self.EMG_fs
        if requires_spike_timing == True:
            behave_dict['spike_timing'] = []
            for i, s in enumerate(self.spikes):
                s = s/30000
                idx = np.where((s>t[0]) & (s<t[1]))[0]
                behave_dict['spike_timing'].append(s[idx])
        if behave_label == 'pg':
            behave_dict['FSR_data'] = seg_fsrs
        return behave_dict                     
            
    def get_all_data_segment(self, requires_raw_EMG = False, requires_spike_timing = False):
        #self.read_behavior_tags(xls_path, xls_name)
        if hasattr(self, 'behave_tags') == True:
            tags = self.behave_tags
            behave_seg = []
            for i in range(len(tags['start_time'])):
                seg = self.get_single_data_segment([tags['start_time'][i], tags['end_time'][i]], 
                                                   tags['tag'][i], 
                                                   requires_raw_EMG,
                                                   requires_spike_timing)
                behave_seg.append(seg)
            return behave_seg
        else:
            print('There is no behavior related information in this file, check again')
            return 0
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        