import numpy as np
import _pickle as pickle
import time
import getpass
import joblib
import os
import copy
import xlrd
import fnmatch
from load_intan_rhd_format import read_data
from scipy import stats, signal
from intanutil.notch_filter import notch_filter
from brpylib import NevFile, NsxFile
from scipy.signal import argrelextrema
from collections import defaultdict
from cage_data_utils import get_paired_EMG_index, find_bad_EMG_index_from_list, delete_paired_bad_channel
from cage_data_utils import parse_nev_header, parse_nsx_header
from cage_data_utils import check_FSR_in_list, check_video_sync_in_list, check_EMG_in_list
from cage_data_utils import train_waveform_classifier
from cage_data_utils import parse_bento_annot
from cage_data_utils import find_force_onset
from cage_data_utils import validate_sync_pulse
from cage_data_utils import read_video_timeframe_from_txt

# sklearn stuff for the wiener filter etc
from sklearn import linear_model, model_selection, metrics
from scipy.optimize import least_squares

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
    
class cage_data:
    def __init__(self):
        self.date_num = 0
        self.has_EMG = 0
        self.meta = dict()
        self.meta['Processes by'] = getpass.getuser()
        self.meta['Processes at'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('An empty cage_data object has been created.')
        
    # --------------------------------         
    def create(self,  nev_file, path = None, rhd_file = '', rhd_path = None,   
               is_sorted = 0, empty_channels = [],      
               bad_EMG = [], do_notch = 1, comb_filter = 0,
               mot_file = '', mot_path = None):                          
        """
        'nev_mat_file' is the neural data filename,
        'rhd_file' is the EMG data filename
        'mot_file' is the motion tracking filename
        
        Required data files. Set 'rhd_file' as ' ' if no wireless EMG recordings
        Is the nev file sorted? Any empty channel?
        Whether apply notch filter? Should remove bad EMG channels?
        Comb filter for EMG preprocessing                                                    
        .mot file if doing motion tracking


        if 'path' is specified, it will look for the .nev inside of the path.
        Otherwise it will first look to see if the file name contains a path, and if
        not will look inside of the current working directory.

        if 'rhd_path' is specified, it will look for the .rhd at that location.
        backup order: specified full path inside of file name, located in the same
        spot as the nev file, and finally in the current working directory.

        if 'rhd_path' is specified, it will look for the .rhd at that location.
        backup order: specified full path inside of file name, located in the same
        spot as the nev file, and finally in the current working directory.
        

        """
        self.is_sorted = is_sorted
        

        # path parsing -- allows .rhd, .mot, and .nev files in different locations
        # plus allows you to use the full path name for the 
        if not path: # if the passed path was empty
            path,nev_file = os.path.split(nev_file) # pull the path from the filename
        if rhd_file and not rhd_path: # if the passed rhd path was empty
            rhd_path,rhd_file = os.path.split(rhd_file) # pull the path from the filename
            if not rhd_path: # if that doesn't work
                rhd_path = path # pull it from the nev file
        if mot_file and not mot_path:
            mot_path,mot_file = os.path.split(mot_file)
            if not mot_path:
                mot_path = path
        

        if os.path.splitext(nev_file)[-1] != '.nev': # cleaner than using string manipulation
            nev_file = nev_file + '.nev'
        # -------- Read the nev file -------- #
        print(os.path.join(path,nev_file))
        self.parse_nev_file(os.path.join(path,nev_file), is_sorted, empty_channels)
        # -------- To check whether rhd_file is along with the nev file -------- #
        if rhd_file == '':
            print('No wireless EMGs recorded during this session')
        else:
            self.has_EMG = 1
            self.parse_rhd_file(os.path.join(rhd_path, rhd_file), do_notch, bad_EMG, comb_filter)
            print(self.EMG_names)
        
        # -------- To check if any .nsx file along with the nev file -------- #     
        file_list = fnmatch.filter(os.listdir( path ), os.path.splitext(nev_file)[0] + '.ns*')
        # -------- To check .ns6 file first -------- #
        if os.path.splitext(nev_file)[0] + '.ns6' in file_list:
            self.raw_data = self.parse_ns6_file(path + os.path.splitext(nev_file)[0] + '.ns6')
            print('Raw data (fs = 30kHz) was recorded along with spike data in this session!')
            file_list.remove(os.path.splitext(nev_file)[0] + '.ns6')
        else:
            print('No raw data along with this recording session.')
        # -------- To check .nsx file (ns1 - ns5) -------- #
        nsx_durations = []
        if file_list == []:
            print('There is no nsx file recorded with the nev file!')
            analog_list = []
        else:
            analog_list = []
            for each in file_list:
                data, duration = self.parse_nsx_file(os.path.join(path,each))
                analog_list.append(data)
                nsx_durations.append(duration)
        self.analog = {}
        if analog_list != []:
            FSR_timeframe, FSR_data = check_FSR_in_list(analog_list, lpf = True)
            video_sync_timeframe, video_sync = check_video_sync_in_list(analog_list)
            if FSR_timeframe != []:
                self.analog['FSR_data'] = FSR_data
                self.analog['FSR_data_timeframe'] = FSR_timeframe
            if video_sync_timeframe != []:
                self.analog['video_sync'] = video_sync
                self.analog['video_sync_timeframe'] = video_sync_timeframe
            if self.has_EMG == 0:
                EMG_timeframe, EMG_fs, EMG_diff, EMG_names = check_EMG_in_list(analog_list)
                if EMG_timeframe != []:
                    self.has_EMG = 1
                    self.EMG_names = EMG_names
                    self.EMG_diff = EMG_diff
                    self.EMG_timeframe = EMG_timeframe
                    self.EMG_fs = EMG_fs
   
        if self.has_EMG == 1:
            self.file_length = self.EMG_timeframe[-1]
        elif nsx_durations != []:
            self.file_length = nsx_durations[0]
        else:
            self.file_length = self.nev_duration

        # Several of these will be changed in the next section
        self.is_cortical_cleaned = False
        self.is_EMG_filtered = False
        self.is_data_binned = False
        self.is_spike_smoothed = False



        # -------- Do some simple pre-processing, including basic cortical data cleaning and EMG filtering -------- #
        self.clean_cortical_data()
        if self.has_EMG == 1:
            self.EMG_filtering(10) # Filter the EMG_diff with an LPF (fc = 10 Hz)
        # -------- To check whether there's any annotation file generated by Bento -------- # 
        bento_flag = 0
        file_list = fnmatch.filter(os.listdir( path ), nev_file[:-4] + '.annot')
        if file_list == []:
            print('There is no .annot file')
        else:
            txt_file_list = fnmatch.filter(os.listdir( path ), nev_file[:-4] + '.txt')
            if txt_file_list == []:
                self.read_behavior_tags_bento(path, file_list[0])
            else:
                self.read_behavior_tags_bento_txt(path, file_list[0])
            bento_flag = 1
            print('.annot file found!')
       
        if bento_flag == 0:
            print('No annotations from Bento, try to read excel instead')
            file_list = fnmatch.filter(os.listdir( path ), nev_file[:-4] + '.xlsx')
            if file_list == []:
                print('There is no .xlsx file')
            else:
                self.read_behavior_tags_excel(path, file_list[0])       

        # motion tracking -- from openSim
        if mot_file:
            self.has_mot = 1 # set a flag
            self.parse_mot_file(os.path.join(os.path.join(mot_path,mot_file)))
        else:
            self.has_mot = 0

        # give an update 
        self.pre_processing_summary()
        print('cage_data structure created')
        
        
    # --------------------------------         
    def pre_processing_summary(self):
        if hasattr(self, 'is_sorted'):
            if self.is_sorted == 1:
                print('This is a sorted file')
            else:
                print('This is a non-sorted file')
        if hasattr(self, 'EMG_diff'):
            print('EMG filtered? -- %s' %(self.is_EMG_filtered))
        else:
            print('EMG filtered? -- %s' %('There is no EMG in this recording.'))
        print('Cortical data cleaned? -- %s' %(self.is_cortical_cleaned))
        if hasattr(self, 'is_data_binned'):
            print('Data binned? -- %s' %(self.is_data_binned))
        if hasattr(self, 'is_spike_smoothed'):
            print('Spikes smoothed? -- %s' %(self.is_spike_smoothed))

    # --------------------------------         
    def parse_nev_file(self, file_name, is_sorted, empty_channels):
        time_s = time.time()
        
        NevFileObj = NevFile(file_name)
        basic_header = NevFileObj.basic_header
        extended_header = NevFileObj.extended_headers
        elec_id_list, elec_label_list, thresholds = parse_nev_header(extended_header)
        
        # ----------- Delete empty electrode channels, they may be used as references or other purpose ---------- #
        empty_str = ['elec' + str(each) for each in empty_channels]
        bad_num = [elec_label_list.index(each) for each in empty_str]
        for idx in sorted(bad_num, reverse=True):
            del(elec_label_list[idx])
            del(elec_id_list[idx])
            del(thresholds[idx])
        
        output = NevFileObj.getdata(elec_ids='all')
        NevFileObj.datafile.close()
        
        # ---------- Deal with actual spike data ---------- #
        time_stamp = np.asarray(output['spike_events']['TimeStamps'])
        electrode = np.asarray(output['spike_events']['Channel'])
        unit = np.asarray(output['spike_events']['Unit'])
        waveform = np.asarray(output['spike_events']['Waveforms'])
        
        s_spikes, s_waveforms = [], []
        # -------- Two conditions, one for sorted files, another for unsorted files -------- #
        # -------- Default is unsorted -------- #
        if is_sorted == 0:
            for each in elec_id_list:
                # -------- Read only unsorted waveforms, avoid cases where 'unit == 255'
                idx = np.where((electrode == each) & (unit == 0))[0]
                s_spikes.append(time_stamp[idx])
                s_waveforms.append(waveform[idx,:])
        else:
            # -------- Codes for reading sorted files -------- #
            # -------- Max number of units in each channel is set as '5' --------#
            MAX = 5
            self.sorted_ch_lbl, self.sorted_elec_id, self.sorted_unit, self.sorted_unit_name = [], [], [], []
            for i, each in enumerate(elec_id_list):
                for u in range(1, MAX + 1):
                    idx = np.where((electrode == each) & (unit == u))[0]
                    if len(idx) > 0:
                        s_spikes.append(time_stamp[idx])
                        s_waveforms.append(waveform[idx,:])
                        self.sorted_elec_id.append(each)
                        self.sorted_ch_lbl.append(elec_label_list[i])
                        self.sorted_unit.append(u)
                        self.sorted_unit_name.append(elec_label_list[i] + str(u))
                    else:
                        break
        
        # -------- Give a number of properties values -------- #
        self.date_num = int(''.join(c for c in str(basic_header['TimeOrigin']) if c.isdigit())[:8])
        self.nev_fs = basic_header['SampleTimeResolution']
        self.nev_duration = np.max(time_stamp)/self.nev_fs
        self.electrode_meta = []
        self.thresholds = thresholds
        self.spikes = s_spikes
        self.waveforms = s_waveforms
        self.ch_lbl = elec_label_list
        self.elec_label = elec_label_list
        self.elec_id = elec_id_list
                
        time_e = time.time()
        print('Parsing the nev file took %.3f s'%(time_e - time_s))
    
    # --------------------------------         
    def parse_nsx_file(self, file_name):
        NsxFileObj = NsxFile(file_name)
        header = NsxFileObj.extended_headers
        data = NsxFileObj.getdata(elec_ids='all', start_time_s=0, data_time_s='all', downsample=1)
        NsxFileObj.datafile.close()
        
        analog_label, max_analog, max_digital = parse_nsx_header(header, data)
        analog = {}
        analog['label'] = analog_label
        analog['MaxDigiValue'] = max_digital
        analog['MaxAnalogValue'] = max_analog
        analog['MinDigiValue'] = [-each for each in analog['MaxDigiValue']]
        analog['MinAnalogValue'] = [-each for each in analog['MaxAnalogValue']]
        analog['fs'] = data['samp_per_s']
        analog['data'] = data['data'].T
        analog['time_frame'] = np.arange(len(analog['data']))/analog['fs']
        nsx_duration = data['data_time_s']
        return analog, nsx_duration        
   
    # --------------------------------         
    def parse_ns6_file(self, file_name):
        ns6_file_name = file_name[:-4]+'.ns6'
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
                        raw_data['ch_lbl'].append(self.elec_label[self.elec_id.index(each)])
                    else:
                        raw_data['ch_lbl'].append('No elec label')
            else:
                raw_data['ch_lbl'] = 0
            raw_data['timeframe'] = np.arange(len(raw_data['data']))/raw_data['fs']
            raw_data['elec_label'] = raw_data['ch_lbl']
            self.ns6_duration = _raw_data['data_time_s']
            return raw_data
        
    # --------------------------------         
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
        # ds = int(d1[0] - int((d1[0]-d0[0])*0.2))
        # de = int(d1[-1] + int((d0[-1]-d1[-1])*0.2))
        ds = int(d0[0])
        de = int(d1[-1])
        rhd_timeframe = np.arange(de-ds+1)/self.EMG_fs

        # update the class instance
        self.EMG_names = EMG_names
        self.EMG_diff = list(EMG_diff[:, ds:de])
        self.EMG_timeframe = rhd_timeframe
        
    # --------------------------------         
    def parse_mot_file(self, filename):
        '''
        Bringing in the joint information from OpenSim for the motion tracking
        system

        inputs:
            .mot filename

        outputs:
            tracking location names, tracking locations, timestamps

        This is currently set to bring in .mot files. For information on the 
        setup of .mot files, refer to 
        https://simtk-confluence.stanford.edu:8443/display/OpenSim/Motion+%28.mot%29+Files
        '''

        with open(file=filename, mode='r') as mot_file:
            print('processing .mot file')

            line = mot_file.readline()
            # .mot header v1
            if 'name' in line:
                while 'endheader' not in line:
                    line_parse = line.split("=") # keep going until we're at the end of the header
                    if line_parse[0] == 'datarows':
                        n_rows = int(line_parse[-1])
                    elif line_parse[0] == 'datacolumns':
                        n_cols = int(line_parse[-1])
                    line = mot_file.readline() # read a new line

            # .mot header v2 -- openSim style
            elif 'Coordinates' in line:
                while 'endheader' not in line:
                    line_parse = line.split("=") # keep going until we're at the end of the header
                    if line_parse[0] == 'nRows':
                        n_rows = int(line_parse[-1])
                    elif line_parse[0] == 'nColumns':
                        n_cols = int(line_parse[-1])
                    line = mot_file.readline() # read a new line


            # bring in all of the data
            col_headers = mot_file.readline() # column headers
            data = np.array([line.split('\t') for line in mot_file.readlines()], dtype=float) # the rest of the file parsed into a numpy array

            if data.shape != (n_rows,n_cols):
                print(f"n_rows: {n_rows}, n_cols: {n_cols}, data_shape: {data.shape}")
                print('Warning: Data from .mot table not sized as expected')
            
            # separate timestamps from everything else.

            timestamps = data[:,0]
            track_loc = data[:,1:]

            # check for sync signals
            if 'video_sync' in self.analog.keys():
                offset = self.analog['video_sync_timeframe']\
                    [np.where(np.diff(self.analog['video_sync'])>1000)[0][0]]
                timestamps = timestamps + offset
            else:
                print('No video sync found. mot_data will not be aligned!')

            # put it into the cage_data. 
            self.mot_data = track_loc
            self.mot_timestamps = timestamps
            self.mot_names = col_headers

    # --------------------------------         
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

    # --------------------------------         
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
            
    # --------------------------------         
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
        
        # save it into the structure
        self.binned['timeframe'] = bins[1:]             # times
        self.binned['spikes'] = np.array(binned_spikes).T # spiking firing rates
        self.binned['spikes_labels'] = self.elec_label  # labels


    # --------------------------------         
    def EMG_downsample(self, new_fs):
        if hasattr(self, 'filtered_EMG'):
            self.binned['filtered_EMG_labels'] = self.EMG_names # just copy it on over
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
            self.binned['filtered_EMG'] = down_sampled
            # return down_sampled
        else:
            print('Filter EMG first!')
            return 0
        
    # --------------------------------         
    def FSR_data_downsample(self, new_fs):
        '''
        new_fs      new sampling frequency. 
        '''
        if 'FSR_data' in self.analog.keys():
            if 'FSR_data_timeframe' in self.analog.keys():
                fs = 1/stats.mode(np.diff(self.analog['FSR_data_timeframe']))[0][0]
            elif 'time_frame' in self.analog.keys():
                fs = 1/stats.mode(np.diff(self.analog['time_frame']))[0][0]
            else:
                fs = self.analog['analog_fs']
            down_sampled = []
            n = fs/new_fs
            length = int(np.floor(np.size(self.analog['FSR_data'][0])/n))
            for each in self.analog['FSR_data']:
                temp = []
                for i in range( 1, length ):
                    temp.append(each[int(np.floor(i*n))])
                temp = np.asarray(temp)
                down_sampled.append(temp)
            print('FSR data have been downsampled')
            self.binned['FSR_data'] = down_sampled
            # return down_sampled
        else:
            print('There is no FSR data in this dataset, please check')
            return 0
        
    # --------------------------------         
    def bin_data(self, bin_size, mode = 'center'):
        '''
        bin_size    length of the bin (in seconds)
        mode        binning mode: left, center [default], right

        '''
        if not hasattr(self, 'binned'):
            self.binned = {}
        
        # EMG flags
        if not hasattr(self, 'has_EMG'):
            if hasattr(self, 'EMG_names'):
                self.has_EMG = 1
            else:
                self.has_EMG = 0

        # bin those spikes, save into the class
        self.bin_spikes(bin_size, mode) 


        if self.has_EMG == 1:
            # self.binned['filtered_EMG'] = self.EMG_downsample(1/bin_size)
            self.EMG_downsample(1/bin_size)
            truncated_len = min(len(self.binned['filtered_EMG'][0]), len(self.binned['spikes'][0]))
            for (i, each) in enumerate(self.binned['spikes']):
                self.binned['spikes'][i] = each[:truncated_len]
            for (i, each) in enumerate(self.binned['filtered_EMG']):
                self.binned['filtered_EMG'][i] = each[:truncated_len]
            self.binned['timeframe'] = self.binned['timeframe'][:truncated_len]

        if 'FSR_data' in self.analog: # might have analog but not FSR. making it specific
            # self.binned['FSR_data'] = self.FSR_data_downsample(1/bin_size)
            self.FSR_data_downsample(1/bin_size)
        self.is_data_binned = True

        if self.has_mot: # this should be at the final binning frequency, me thinks
            temp = np.zeros((self.binned['timeframe'].shape[0], self.mot_data.shape[1])) # set up an empty temp array
            offset = np.argmin((self.binned['timeframe']-self.mot_timestamps[0])**2) # find where the data starts
            temp[offset:offset+len(self.mot_data)] = self.mot_data # fill in everything beyond that point
            self.binned['mot_data'] = temp[:len(self.binned['timeframe']),:] # clip off everything that's too long, store it

        print('Data have been binned.')

    # --------------------------------         
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
            
    # --------------------------------         
    def save_to_pickle(self, save_path, file_name):

        # if it doesn't have an extension on the end
        if not os.path.splitext(file_name)[1]:
            file_name = file_name.join('.pkl')
        
        # combine the path and the filename in the appropriate manner
        save_name = os.path.join(save_path, file_name)

        # write it 
        with open (save_name, 'wb') as fp:
            pickle.dump(self, fp)
        print('Save to %s successfully \n' %(save_name))
        
    # --------------------------------         
    def ximea_video_sync(self):
        if 'video_sync' in self.analog.keys():
           if 'video_sync_timeframe' not in self.analog.keys():
               self.analog['video_sync_timeframe'] = self.analog['time_frame']
           sync_pulse = self.analog['video_sync']
           M = np.max(sync_pulse)
           sync_pulse[np.where(sync_pulse<M/3)[0]] = 0
           sync_pulse[np.where(sync_pulse>M/3)[0]] = 32000
           diff_sync_pulse = np.diff(sync_pulse)
           peaks = list(argrelextrema(diff_sync_pulse, np.greater)[0])
           bad_peaks = validate_sync_pulse(sync_pulse, 32000)
           for each in bad_peaks:
               del(peaks[each])
           print('There are %d pulses for video sync in this file'%(len(peaks)))
           video_timestamps = self.analog['video_sync_timeframe'][peaks]
        else:
            print('No video sync signals in this file')
            peaks = 0
            video_timestamps = 0
        return video_timestamps
                                
    # --------------------------------         
    def clean_cortical_data_with_classifier(self, template_file_path, template_file):
        if template_file_path[-1] != '/':
            template_file_path = template_file_path + '/'
        joblib_list = fnmatch.filter(os.listdir( template_file_path ), '*.joblib')
        if joblib_list == []: 
            print('Need to train the classifier first!')
            clf_file = train_waveform_classifier(template_file_path, template_file)
        else:
            print('Classifier already trained!')
            clf_file = template_file_path + joblib_list[0]
        self.bad_waveforms = []
        clf = joblib.load(clf_file)
        for i, each in enumerate(self.waveforms):
            if each.shape[0] == 0:
                continue
            else:
                res = clf.predict(each)
                bad_idx = np.where(res == 1)[0]
                if len(bad_idx) > 0:
                    self.bad_waveforms.append(each[bad_idx, :])
                    self.waveforms[i] = np.delete(self.waveforms[i], bad_idx, axis = 0)
                    self.spikes[i] = np.delete(self.spikes[i], bad_idx)   
    
    # --------------------------------         
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
          
    # --------------------------------         
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

    # --------------------------------         
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
                
    # --------------------------------         
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

    # --------------------------------         
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

    # --------------------------------         
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
            del(self.elec_label[idx])
    
    # --------------------------------         
    def find_pg_force_onset(self, ch, thr = 0.4):
        if 'FSR_data' in self.analog.keys():
            ft = self.analog['FSR_data_timeframe']
            f = self.analog['FSR_data']
            pg_idx = [i for i, each in enumerate(self.behave_tags['tag']) if each == 'pg']
            pg_start_time = [self.behave_tags['start_time'][i]-0.1 for i in pg_idx]
            pg_end_time = [self.behave_tags['end_time'][i] for i in pg_idx]
            pg_trial_idx = [np.where( (ft>pg_start_time[i])&(ft<pg_end_time[i]) )[0] for i in range(len(pg_start_time))]
            pg_trial_force = [np.vstack((f[0][idx], f[1][idx])).T for idx in pg_trial_idx]
            pg_trial_timeframe = [ft[idx] for idx in pg_trial_idx]
                    
            idx_onset = find_force_onset(pg_trial_force, ch, thr)
            time_onset = [pg_trial_timeframe[i][idx_onset[i]] for i in range(len(pg_trial_timeframe))]
            print('Get the force onset time!')
            return time_onset
        else:
            print('No FSR data in this file')
            return []
    
    # --------------------------------         
    def read_behavior_tags_bento(self, path, file_name):
        self.behave_event = {}
        self.behave_tags = {'tag':[], 'start_time': [], 'end_time': []}
        behave_frame = parse_bento_annot(path, file_name)
        video_timeframe = self.ximea_video_sync()
        if 'bar_touch' in behave_frame.keys(): 
            bar_touch = behave_frame.pop('bar_touch')
            self.behave_event['bar_touch'] = list(video_timeframe[ bar_touch[:, 0] ])
        if 'treat_touch' in behave_frame.keys():
            treat_touch = behave_frame.pop('treat_touch')
            self.behave_event['treat_touch'] = list(video_timeframe[ treat_touch[:, 0] ])
        if behave_frame != []:
            for key,value in behave_frame.items():
                for i in range(len(value)):
                    self.behave_tags['tag'].append(key)
                    self.behave_tags['start_time'].append( video_timeframe[value[i, 0]] )
                    self.behave_tags['end_time'].append( video_timeframe[value[i, 1]] )
        if 'pg' in behave_frame.keys():
            self.behave_event['pg_force_onset'] = self.find_pg_force_onset(0, 0.4)
            
    # --------------------------------         
    def read_behavior_tags_bento_txt(self, path, file_name):
        self.behave_event = {}
        self.behave_tags = {'tag':[], 'start_time': [], 'end_time': []}
        behave_frame = parse_bento_annot(path, file_name)
        txt_file_name = file_name[:-6] + '.txt'
        video_timeframe = read_video_timeframe_from_txt(path, txt_file_name)
        if 'bar_touch' in behave_frame.keys(): 
            bar_touch = behave_frame.pop('bar_touch')
            self.behave_event['bar_touch'] = list(video_timeframe[ bar_touch[:, 0]-1 ])
        if 'treat_touch' in behave_frame.keys():
            treat_touch = behave_frame.pop('treat_touch')
            self.behave_event['treat_touch'] = list(video_timeframe[ treat_touch[:, 0]-1 ])
        if behave_frame != []:
            for key,value in behave_frame.items():
                for i in range(len(value)):
                    self.behave_tags['tag'].append(key)
                    self.behave_tags['start_time'].append( video_timeframe[ value[i, 0]-1 ] )
                    self.behave_tags['end_time'].append( video_timeframe[ value[i, 1]-1 ] )
        if 'pg' in behave_frame.keys():
            self.behave_event['pg_force_onset'] = self.find_pg_force_onset(0, 0.4)
            
    # --------------------------------         
    def read_behavior_tags_excel(self, path, file_name):
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
            self.behave_event = dict()
            self.behave_tags['start_time'] = list(video_timeframe[start])
            self.behave_tags['end_time'] = list(video_timeframe[ends])
            self.behave_tags['tag'] = tags
        if 'pg' in self.behave_tags['tag']:
           self.behave_event['pg_force_onset'] = self.find_pg_force_onset(0, 0.4) 
            
    # --------------------------------         
    def get_behave_segment(self, name, time1, time2, requires_raw_EMG = False, requires_spike_timing = False, requires_30k = False):
        # -------- determine if the name is from behave tags or behave events -------- #
        if hasattr(self, 'behave_event'):
            if name in self.behave_event.keys():
                t = np.asarray(self.behave_event[name])
                t1, t2 = t-time1, t+time2
            else:
                idx = [i for i, each in enumerate(self.behave_tags['tag']) if each == name]
                t1 = np.asarray([self.behave_tags['start_time'][i] for i in idx])
                t2 = np.asarray([self.behave_tags['end_time'][i] for i in idx])
        else:
            idx = [i for i, each in enumerate(self.behave_tags['tag']) if each == name]
            t1 = np.asarray([self.behave_tags['start_time'][i] for i in idx])
            t2 = np.asarray([self.behave_tags['end_time'][i] for i in idx])
        if hasattr(self,'binned') == 0:
            print('bin the data first!')
        else:
            timeframe = self.binned['timeframe']
            binned_spikes = np.asarray(self.binned['spikes']).T
            emgs = np.asarray(self.binned['filtered_EMG']).T
            if 'FSR_data' in self.analog.keys():
                fsrs = np.asarray( self.binned['FSR_data'] ).T
            behave_dict_all = []
            idx = [np.where( (timeframe>each[0]) & (timeframe<each[1]) )[0] for each in zip(t1, t2)]
            for i, each in enumerate(idx):
                behave_dict = {}
                behave_dict['spikes'] = binned_spikes[each, :] 
                behave_dict['EMG'] = emgs[each, :]
                if (name == 'pg')|(name == 'pg_force_onset'):
                    behave_dict['FSR_data'] = fsrs[each, :]
                behave_dict['timeframe'] = timeframe[each] 
                behave_dict['EMG_names'] = self.EMG_names
                behave_dict['unit_names'] = self.elec_label
                behave_dict['label'] = name
                if requires_raw_EMG == True:
                   idx_raw_EMG = np.where((self.EMG_timeframe>t1[i]) & (self.EMG_timeframe<t2[i]))[0]
                   behave_dict['raw_EMG'] = np.asarray(self.EMG_diff).T[idx_raw_EMG, :]
                   behave_dict['raw_EMG_timeframe'] = self.EMG_timeframe[idx_raw_EMG]
                   behave_dict['raw_EMG_fs'] = self.EMG_fs 
                if requires_spike_timing == True:
                   behave_dict['spike_timing'] = []
                   for s in self.spikes:
                       s = s/30000
                       idx_spike_timing = np.where((s>t1[i]) & (s<t2[i]))[0]
                       behave_dict['spike_timing'].append(s[idx_spike_timing] - t1[i])
                if requires_30k == True:
                    timeframe_30k = self.raw_data['timeframe']
                    idx_30k = np.where((timeframe_30k>t1[i]) & (timeframe_30k<t2[i]))[0]
                    behave_dict['30k'] = self.raw_data['data'][idx_30k, :3]
                behave_dict_all.append(behave_dict)
        return behave_dict_all

    # --------------------------------         
    # Wiener filter -- allows for basic non-linearities 
    def filter_builder(self, out_type='EMG', out_labels=None, n_lags=5, nonlinearity=None, train_size=.9):
        '''
        Builds a linear filter between binned threshold crossings and the 
        listed outputs. 

        INPUTS
        out_type:           type of data to predict
        out_labels:         which of those to predict (ie FCRu, Fx, or PIP_tra1)
        nonlinearity:       None, 'poly' for polynomial, or 'exp' for exponential

        OUTPUTS
        mdl:                sklearn.Linear_Model for the linear prediction
        [mdl_nonlin]:       array of coefficients for a nonlinearity, one for each predicted signal. 
                                only returned if nonlinearity is not None
        vaf_train:          Training VAF (metrics.r2_score)
        vaf_test:           Testing VAF (metrics.r2_score)

        '''

        # input flag parsing
        if out_type == 'EMG':
            out_type = 'filtered_EMG'
        if out_type == 'mot':
            out_type = 'mot_data'
        if out_type is None:
            out_type = 'filtered_EMG'

        # check that the desired "train-on" set is available
        binned_list = [key for key in self.binned.keys() if key not in ['timeframe','spikes']]
        if out_type not in binned_list:
            print(f"{out_type} not in binned data. This caged_data has only {binned_list}. Check for typos!")
            return -1

        # make sure the linearity flag is correct
        if nonlinearity not in [None, 'poly', 'Poly', 'exp', 'Exp']:
            if nonlinearity in ['Poly','Polynomial','polynomial']:
                nonlinearity = 'poly'
            elif nonlinearity in ['Exp','exponential','Exponential']:
                nonlinearity = 'exp'
            else:
                print(f"nonlinearity value {nonlinearity} is not supported")
                return -1

        # split into train/test sets based on the percentage given
        mdl = linear_model.LinearRegression()
        train_neur, test_neur, train_target, test_target = model_selection.train_test_split(self.binned['spikes'],self.binned[out_type], train_size=train_size)

        # add lags to the training and testing inputs
        wiener_train = np.zeros((train_neur.shape[0],train_neur.shape[1]*n_lags))
        wiener_test = np.zeros((test_neur.shape[0],test_neur.shape[1]*n_lags))
        for ii in np.arange(n_lags):
            ind_start = ii*train_neur.shape[1]
            ind_end = (ii+1)*train_neur.shape[1]
            wiener_train[ii:,ind_start:ind_end] = train_neur[:ii] # training set
            wiener_test[ii:,ind_start:ind_end] = test_neur[:ii] # test set


        # build the model
        mdl.fit(x=wiener_train, y=train_target)

        # predictions on testing set
        train_pred = mdl.predict(wiener_train)
        test_pred = mdl.predict(wiener_test)

        # spit out the VAFs on the test set
        if not nonlinearity:
            train_vafs = metrics.explained_variance_score(train_target, train_pred, multioutput='raw_values')
            test_vafs = metrics.explained_variance_score(test_target, test_pred, multioutput='raw_values')
            return mdl, train_vafs, test_vafs
        
        # train any non-linearity
        mdl_nonlin = []
        # nonlin_pred = np.zeros(train_pred.shape)

        for ii in np.arange(train_pred.shape[1]): # for each individual signal
            mdl_nonlin[:,ii] = least_squares(self.non_linearity_residuals,\
                 [0.1, 0.1, 0.1], args=(train_pred[:,ii],train_target[:,ii], nonlinearity)).x
            train_pred_nonlin = self.non_linearity(mdl_nonlin[:,ii], train_pred[:,ii], nonlinear_type=nonlinearity)
            test_pred_nonlin = self.non_linearity(mdl_nonlin[:,ii], test_pred[:,ii], nonlinear_type=nonlinearity)
            train_vafs = metrics.explained_variance_score(train_target[:,ii], train_pred_nonlin, multioutput='raw_values')
            test_vafs = metrics.explained_variance_score(test_target[:,ii], test_pred_nonlin, multioutput='raw_values')


        return mdl, mdl_nonlin, train_vafs, test_vafs
    

    # using scipy's least_squares:
    def non_linearity(coeff, y_train, nonlinear_type):
        """
        calculates nonlinear predictions for either polynomial or exponential nonlinearities
        """
        if nonlinear_type == 'poly':
            return coeff[0] + coeff[1]*y_train + coeff[2]*y_train**2
        elif nonlinear_type == 'exponential':
            return coeff[0]*np.exp(coeff[1]*y_train) + coeff[2]
        else:
            return -1

    def non_linearity_residuals(coeff, y_pred, y_act, nonlinear_type):
        if nonlinear_type == 'poly':
            return y_act - (coeff[0] + coeff[1]*y_pred + coeff[2]*y_pred**2)
        elif nonlinear_type == 'exponential':
            return y_act - (coeff[0]*np.exp(coeff[1]*y_pred) + coeff[2])
        else:
            return -1

