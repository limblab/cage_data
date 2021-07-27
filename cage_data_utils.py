import numpy as np
import copy
import fnmatch
from load_intan_rhd_format import read_data
from scipy import stats, signal
from brpylib import NevFile, NsxFile
from scipy.signal import find_peaks
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import random
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
from scipy.signal import argrelextrema

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

def parse_nev_header(header):
    elec_id_list, elec_label_list = [], []
    for each in header:
        if each['PacketID'] == 'NEUEVLBL':
            if each['Label'][:4] == 'elec':
                elec_id_list.append(each['ElectrodeID'])
                elec_label_list.append(each['Label'])
        else:
            continue
        
    threshold = np.zeros((np.size(elec_id_list, 0)))
    for each in header:
        if each['PacketID'] == 'NEUEVWAV':
            if each['ElectrodeID'] <= np.max(elec_id_list):
                threshold[each['ElectrodeID']-1] = each['LowThreshold']
            else:
                break
        else:
            continue
    
    return elec_id_list, elec_label_list, list(threshold)

def parse_nsx_header(header, data):
    idx = data['ExtendedHeaderIndices']
    analog_label = [header[i]['ElectrodeLabel'] for i in idx]
    MaxAnalogValue = [header[i]['MaxAnalogValue'] for i in idx]
    MaxDigitalValue = [header[i]['MaxDigitalValue'] for i in idx]
    return analog_label, MaxAnalogValue, MaxDigitalValue

def check_FSR_in_list(analog_list, lpf = True):
    """
    To check whether any FSR data inside the list obtained from reading nsx files

    Parameters
    ----------
    analog_list: List,
        DESCRIPTION. The list containing analog data read from the nsx files
    lpf : Logic, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    The FSR data extracted from the list

    """
    for n, analog in enumerate(analog_list):
        idx = []
        for j, label in enumerate(analog['label']):
            if 'FSR' in label:
                idx.append(j)
        if idx != []:
            break
    if idx != []:
        FSR_data = []
        if lpf == True:
            blow, alow = signal.butter(4, 10/(analog_list[n]['fs']/2), 'low')
        for i in idx:
            temp = analog_list[n]['data'][:, i]
            temp = (temp-analog_list[n]['MinDigiValue'][i])/32768*analog_list[n]['MaxAnalogValue'][i]/1000 
            if lpf == True:
                FSR_data.append(signal.filtfilt(blow, alow, temp))
            else:
                FSR_data.append(temp)
        print('This recording session contains FSR data.')
        return analog_list[n]['time_frame'], FSR_data
    else:
        print('There is no FSR data in this recording session!')
        return [], []
    
def check_video_sync_in_list(analog_list):
    m = []
    for n, analog in enumerate(analog_list):
        for j, label in enumerate(analog['label']):
            if ('video_sync' in label)|('kinectSync' in label)|('videosync' in label):
                m = j
                break
        if m != []:
            break
    if m != []:
        video_sync = analog_list[n]['data'][:, m]
        print('This recording session contains video sync pulses.')
        return analog_list[n]['time_frame'], video_sync
    else:
        print('There is no video sync pulses in this recording session!')
        return [], []  
    
def check_EMG_in_list(analog_list):
    for n, analog in enumerate(analog_list):
        idx = []
        for j, label in enumerate(analog['label']):
            if 'EMG' in label:
                idx.append(j)
        if idx != []:
            break
    if idx != []:
        EMG_names, EMG_diff = [], []
        for i in idx:
            EMG_names.append(analog_list[n]['label'][i])
            temp = analog_list[n]['data'][:, i]
            EMG_diff.append(temp) 
        print('This recording session contains EMG data recorded by Cerebus.')
        return analog_list[n]['time_frame'], analog_list[n]['fs'], EMG_diff, EMG_names
    else:
        print('There is no EMG data in this session recorded by Cerebus')
        return [], [], [], [] 

def train_waveform_classifier(path, file_name):
    if path[-1] != '/':
        path = path + '/'
    NevFileObj = NevFile(path + file_name)
    output = NevFileObj.getdata(elec_ids='all')
    NevFileObj.datafile.close()
    # ---------- Deal with actual spike data ---------- #
    unit = np.asarray(output['spike_events']['Unit'])
    waveform = np.asarray(output['spike_events']['Waveforms'])
    invalid_idx = np.where(unit == 255)[0]
    good_idx = np.where(unit == 1)[0]
    invalid_waveform = np.array([waveform[i, :] for i in invalid_idx])
    good_waveform = np.array([waveform[i, :] for i in good_idx])
    invalid_label = np.ones((invalid_waveform.shape[0], ))
    good_label = np.zeros((good_waveform.shape[0], ))
    
    N = min(good_waveform.shape[0], invalid_waveform.shape[0])
    idx1 = random.sample(range(0, invalid_waveform.shape[0]), int(N))
    idx2 = random.sample(range(0, good_waveform.shape[0]), int(N))
    x = np.concatenate( (invalid_waveform[idx1, :], good_waveform[idx2, :]) )
    y = np.concatenate( (invalid_label[idx1], good_label[idx2]) )
    split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for train_index, test_index in split.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index] 
    # -------- Train classifier -------- #        
    rf_clf = RandomForestClassifier(max_depth = 100, random_state = 0, n_estimators = 150)
    rf_clf.fit(x_train, y_train)
    y_pred = rf_clf.predict(x_test)
    score = rf_clf.score(x_test, y_test)    
    joblib.dump(rf_clf, path + 'rf_' + file_name[:-4] + '.joblib')
    print('Classifier saved, the accuracy on test set is %.3f'%(score))
    return path + 'rf_' + file_name[:-4] + '.joblib'

def parse_bento_annot(path, file_name):
    if path[-1] != '/':
        path = path+'/'
    with open(path + file_name, 'r', encoding = 'utf-8') as f:
        contents = f.readlines()
    s = contents.index('List of annotations:\n')
    i = 0
    while contents[s+i] != '\n':
        i = i+1
    behavior_list = [contents[s+k] for k in range(1, i)]    
    behavior = {}
    for each in behavior_list: 
        try:
            b = contents.index('>'+each)
        except Exception:
            continue
        i = 2
        start, stop = [], []
        while contents[b+i] != '\n':
            nums = re.findall('\d*', contents[b+i])
            start.append(int(nums[0]))
            stop.append(int(nums[2]))
            i = i+1
        behavior[each[:-1]] = np.vstack((start, stop)).T
    return behavior
            
def find_force_onset(force_list, ch, thr):
    onset_num = []
    for each in force_list:
        f = each[:, ch]#np.sqrt(each[:, 0]**2 + each[:, 1]**2)
        df = np.diff(f)
        temp = np.where(df >= thr*np.max(df))[0]              
        if len(temp) == 0:
            onset_num.append(0)
        else:
            onset_num.append(temp[0])
    return onset_num      
    
def validate_sync_pulse(sync_pulse, M):
    diff_sync_pulse = np.diff(sync_pulse)
    peaks = list(argrelextrema(diff_sync_pulse, np.greater)[0]+1)
    bad_idx = []
    for i, each in enumerate(peaks):
        if len(np.where( sync_pulse[each-11:each] != 0 )[0]) > 0:
            bad_idx.append(i)
    return sorted(bad_idx, reverse = True)
        
                
                
            

















