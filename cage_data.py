import numpy as np
import _pickle as pickle
import time
import h5py
import getpass
import matplotlib.pyplot as plt
import joblib
from load_intan_rhd_format import read_data
from scipy import stats, signal
from intanutil.notch_filter import notch_filter
from brpylib import NevFile 
from scipy.signal import find_peaks

memo1 = """Since Blackrock Python codes (brpy) are too slow when reading .nev files, we use MATLAB version of .nev files instead."""
memo2 = """Please make sure MATLAB version of .nev files are in your target directory. """
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
               has_analog = 0):                         # Analog files are for sync pulses of videos.
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
            self.nev_mat_file = ''.join((path, '/', nev_mat_file))
            if self.has_EMG == 1:
                self.rhd_file = ''.join((path, '/', rhd_file))
        self.parse_nev_mat_file(self.nev_mat_file, is_sorted, empty_channels, has_analog)
        if self.has_EMG == 1:
            try:
                self.date_num = int(rhd_file[:8])
                self.date_num = 0
            except ValueError:
                print('Check the file name of the .rhd file!')
            else:
                pass

            self.EMG_names, self.EMG_diff, self.EMG_timeframe = self.parse_rhd_file(self.rhd_file, 
                                                                                    do_notch, bad_EMG)
            self.file_length = self.EMG_timeframe[-1]
            print(self.EMG_names)
        else:
            self.file_length = self.nev_duration
        
        self.is_cortical_cleaned = False
        self.is_EMG_filtered = False
        self.is_data_binned = False
        self.is_spike_smoothed = False
        self.binned = {}
        self.pre_processing_summary()
        
    def pre_processing_summary(self):
        if hasattr(self, 'is_sorted'):
            if self.is_sorted == 1:
                print('This is a sorted file')
            else:
                print('This is a non-sorted file')
        if hasattr(self, 'has_EMG'):
            if self.has_EMG == 1:        
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
        nev_file = h5py.File(filename)['NEV']
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
        
        if has_analog == 1:
           self.analog = {}
           self.analog['analog_fs'] = nev_file['analog_fs'][0][0]
           self.analog['analog_data'] = np.asarray(nev_file['analog_data'])
           analog_lbl = list(np.asarray(nev_file['analog_labels']).T)
           for (i, each) in enumerate(analog_lbl):
               analog_lbl[i] = ''.join(chr(x) for x in each).strip(b'\x00'.decode())
           self.analog['analog_lbl'] = analog_lbl
           self.analog['analog_MaxDigiValue'] = nev_file['analog_MaxDigiValue'][0][0]
           self.analog['analog_MaxAnalogValue'] = nev_file['analog_MaxAnalogValue'][0][0]
           self.analog['analog_MinDigiValue'] = -self.analog['analog_MaxDigiValue']
           self.analog['analog_MinAnalogValue'] = -self.analog['analog_MaxAnalogValue']
           if 'video_sync' in self.analog['analog_lbl']:
               self.analog['video_sync'] = self.analog['analog_data'][:, self.analog['analog_lbl'].index('video_sync')]
           elif 'kinectSync' in self.analog['analog_lbl']:
               self.analog['video_sync'] = self.analog['analog_data'][:, self.analog['analog_lbl'].index('kinectSync')]
           else:
               print('No sync pulses!')
        
        e = time.clock()
        print("%.3f s for parsing the nev-mat file" %(e-s))
        
    def parse_rhd_file(self, filename, notch, bad_EMG):
        rhd_data = read_data(filename)
        if self.date_num < 20190701:
            self.EMG_fs = 2011.148
        else:
            self.EMG_fs = rhd_data['frequency_parameters']['amplifier_sample_rate']
        EMG_single = rhd_data['amplifier_data']
        
        EMG_names_single = []
        for each in rhd_data['amplifier_channels']:
            EMG_names_single.append(each['custom_channel_name'])
        # ---------- To get paired EMG channels for software diffrence ---------- #
        EMG_names, EMG_index1, EMG_index2 = self.get_paired_EMG_index(EMG_names_single)

        EMG_diff = []
        for i in range(len(EMG_index1)):
            EMG_diff.append(EMG_single[EMG_index1[i], :] - EMG_single[EMG_index2[i], :])
        
        # ---------- Based on the list in bad_EMG, substitute some channels with single end EMG ---------- #
        if bad_EMG:
            bad_idx, paired_idx = self.find_bad_EMG_index_from_list(EMG_names_single, bad_EMG)
            for (i,each) in enumerate(bad_EMG):
                target_idx = EMG_names.index(each[:-2])
                EMG_diff[target_idx] = EMG_single[paired_idx[i], :]
                print("For noisy channel %s, use only one single end channel." %(each[:-2]))
                lost_idx = np.where(EMG_diff[target_idx]<-6300)[0]
                if lost_idx.size > 0:
                    EMG_diff[target_idx][lost_idx] = EMG_diff[target_idx][lost_idx[0]-10]
        
        # ---------- Apply notch filter on EMG_diff ---------- #
        if notch == 1:
           print('Applying notch filter.')
           bnotch, anotch = signal.butter(4, [55/(self.EMG_fs/2), 65/(self.EMG_fs/2)], btype='bandstop')
           for (i, each) in enumerate(EMG_diff): 
               EMG_diff[i] = signal.filtfilt(bnotch, anotch, each)
        else:
            print('No notch filter is applied.')
        
        EMG_diff = np.asarray(EMG_diff)
        
        sync_line0 = rhd_data['board_dig_in_data'][0]
        sync_line1 = rhd_data['board_dig_in_data'][1]
        d0 = np.where(sync_line0 == True)[0]
        d1 = np.where(sync_line1 == True)[0]
        ds = int(d1[0] - int((d1[0]-d0[0])*0.2))
        de = int(d1[-1] + int((d0[-1]-d1[-1])*0.2))
        rhd_timeframe = np.arange(de-ds+1)/self.EMG_fs
        return EMG_names, list(EMG_diff[:, ds:de]), rhd_timeframe
    
    def get_paired_EMG_index(self, EMG_names_single):
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
    
    def find_bad_EMG_index_from_list(self, EMG_names_single, bad_EMG):
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
        bins = np.arange(bin_start, self.file_length, bin_size)
        return bins[1:], binned_spikes        
      
    def EMG_downsample(self, new_fs):
        if hasattr(self, 'filtered_EMG'):
            down_sampled = []
            n = self.EMG_fs/new_fs
            length = int(np.floor(np.size(self.filtered_EMG[0])/n)+1)
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
            save_name = ''.join((save_path, file_name, '.pkl'))
        else:
            save_name = ''.join((save_path, '/', file_name, '.pkl'))
        with open (save_name, 'wb') as fp:
            pickle.dump(self, fp)
        print('Save to %s successfully' %(save_name))
        
    def ximea_video_sync(self):
        if hasattr(self, 'analog'):
           sync_pulse = self.analog['video_sync']
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

    def plot_bad_waveforms(self, N):
        if N >= 0 | N < 100: 
            plt.figure(0, figsize = (6,6))
            ax = plt.subplot(111)
            ax.axis('off')
            plt.plot(self.bad_waveforms[N][:, :].T, 'gray')
            plt.title(self.ch_lbl[N] +': '+ str( np.size(self.bad_waveforms[N], 0) ) + ' bad')            
        else:
            print('Wrong number')
        
    def plot_good_waveforms(self, N, plot_avg = 0):
        if N >= 0 | N < 100: 
            plt.figure(0, figsize = (6,6))
            ax = plt.subplot(111)
            ax.axis('off')
            plt.plot(self.waveforms[N][:, :].T, 'b', alpha = 0.5)
            if plot_avg == 1:
                plt.plot(np.mean(self.waveforms[N][: , :].T, axis = 1), 'k')
            plt.title(self.ch_lbl[N] +': '+ str( np.size(self.waveforms[N], 0) ) + ' good')            
        else:
            print('Wrong number')        
        
        
        
        
        
        