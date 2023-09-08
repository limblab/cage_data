import numpy as np
import xlwt
import fnmatch, os
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import signal
import seaborn as sns
import itertools

rcParams['font.family'] = 'Arial'

def get_elec_idx(my_cage_data, elec_num):
    """
    To get the idx of electrodes specified by elec_num
    my_cage_data: cage_data structure
    elec_num: a list containing the number of bad channels
    """
    idx = []
    for each in elec_num:
        if 'elec'+str(each) in my_cage_data.ch_lbl:
            temp = my_cage_data.ch_lbl.index('elec'+str(each))
            idx.append(temp)
    return idx

def del_bad_chs(my_cage_data, elec_num):
    """
    To get rid of everything about the bad channels from my_cage_data
    """
    idx = get_elec_idx(my_cage_data, elec_num)
    for idx in sorted(idx, reverse=True):
        del(my_cage_data.ch_lbl[idx])
        del(my_cage_data.elec_id[idx])
        del(my_cage_data.thresholds[idx])
        del(my_cage_data.waveforms[idx])
        del(my_cage_data.spikes[idx])
#        del(my_cage_data.electrode_meta['elec_pin'][idx])
#        del(my_cage_data.electrode_meta['elec_bank'][idx])
    return my_cage_data

def plot_spike_waveforms_cage_data_structure(my_cage_data, plot_N):    
    R, C = 10, 10
    grid = plt.GridSpec(R,C,wspace=0.1,hspace=0.1)
    for i in range(R):
        for j in range(C):
            ax = plt.subplot(grid[i, j])
            if i*10+j < len(my_cage_data.waveforms):
                ax.plot(my_cage_data.waveforms[i*10+j][:plot_N].T, 'gray')
                ax.text(0, ax.get_ylim()[1], 
                        my_cage_data.ch_lbl[i*10+j] + ' '+ str(my_cage_data.elec_id[i*10+j]-1),
                        fontsize=14)
            ax.axis('off')

def show_waveforms_after_clean(my_cage_data, my_type, N, plot_avg = 0):
    """
    This function is designed to plot the waveforms eliminated by the semi-supervised algorithm
    N is the label of the electrode, not the id
    my_type tells 'good' or 'bad'
    All waveforms will be plotted in 10 subplots in order to be shown more clearly
    """
    try:
        idx = my_cage_data.ch_lbl.index('elec'+str(N))
    except ValueError:
        print('This channel is not in the list')
        return 0
    else:
        pass
    
    try:
        if my_type == 'bad':
            w = my_cage_data.bad_waveforms[idx]
        elif my_type == 'good':
            w = my_cage_data.waveforms[idx]
    except ValueError:
        print('No bad waveforms at this channel')
        return 0
    else:
        pass
    
    plt.figure(my_type+' waveforms on elec%d'%(N), figsize = (18,10))
    plt.suptitle(my_cage_data.ch_lbl[idx]+': %d bad and %d good. '%(np.size(my_cage_data.bad_waveforms[idx], 0), np.size(my_cage_data.waveforms[idx], 0)), fontsize = 16)    
    grid = plt.GridSpec(nrows=5, ncols=8, wspace=0.1, hspace=0.1)
    if len(w)!=0:
        m = len(w)//40
        if m >= 1:
            for i in range(5):
                for j in range(8):
                     ax = plt.subplot(grid[i, j])
                     ax.axis('off')
                     pw = w[(i*8+j)*m:(i*8+j)*m+m , :].T
                     if my_type == 'bad':
                         plt.plot(pw, 'gray')
                     elif my_type == 'good':
                         plt.plot(pw, 'g')
                     if plot_avg == 1:
                         plt.plot(np.mean(pw, axis = 1), 'k')
                     ymin, ymax = ax.get_ylim()
                     plt.text(0, ymax-20,'[%s ~ %s)' %((i*8+j)*m, (i*8+j)*m+m), fontsize = 14,
                              verticalalignment="top",horizontalalignment="left")
        else:
            print('Very few waveforms on this channel')
    else:
        print('No waveform at this channel')
    
def generate_waveform_report(my_cage_data, base_path = './'):
    """
    my_cage_data : the data object
    """
    value_title = ["electrode label", "all", "good", "bad", "bad ratio (%)"]
    bad_n, good_n = [], []
    for (x, y) in zip(my_cage_data.bad_waveforms, my_cage_data.waveforms):
        bad_n.append(len(x))
        good_n.append(len(y))
    all_n = list(np.sum([bad_n, good_n], axis = 0))
    bad_ratio = list(np.asarray(bad_n)/np.asarray(all_n)*100)
    value = np.asarray([all_n, good_n, bad_n, bad_ratio]).T
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('waveform report')
    for i, each in enumerate(value_title):
        sheet.write(0, i, each)
    for i in range(value.shape[0]):
        sheet.write(i+1, 0, my_cage_data.ch_lbl[i])
        for j in range(value.shape[1]):
            sheet.write(i+1, j+1, value[i, j])
    file_name = 'waveforms' + my_cage_data.nev_mat_file[-7:-4]
    workbook.save(base_path + file_name + '.xls')
    print('Report successfully generated!')
    
def pull_data_from_behave_seg_all(behave_seg_all, cat_str, EMG_idx, scale = 0):
    spike, emg, force = [], [], []
    for behave_seg in behave_seg_all:
        for each in behave_seg:
            if each['label'] == cat_str:
                spike.append(each['spike'])
                if scale != 0:
                    emg.append(each['EMG'][:, EMG_idx]/10)
                else:
                    emg.append(each['EMG'][:, EMG_idx])
                if each['label'] == 'pg':
                    force.append(each['FSR_data'])             
    if cat_str == 'pg':
        return spike, emg, force
    else:
        return spike, emg    

def load_obj(name, path):
    with open(path + name, 'rb') as f:
        return pickle.load(f)
    
def load_behave_segs(base_path):
    file_list = fnmatch.filter(os.listdir(base_path), "*.pkl")
    file_list = np.sort(file_list)   
    behave_seg_all = []
    for i in range(len(file_list)):
        behave_seg_all.append(load_obj(file_list[i], base_path))
        print(file_list[i])
    return behave_seg_all
    
def transform_list_to_behave_dict(data_spike, data_EMG, data_spike_timing, EMG_names, raw_EMG = 0, raw_EMG_fs = 0):
    data_dict = dict()
    data_dict['spike'] = data_spike
    data_dict['EMG'] = data_EMG
    data_dict['EMG_names'] = EMG_names
    data_dict['label'] = ''
    data_dict['spike_timing'] = data_spike_timing
    if raw_EMG.any() == True:
        data_dict['raw_EMG'] = raw_EMG
        data_dict['raw_EMG_fs'] = raw_EMG_fs
    return data_dict
    
def get_time_ticks(x):
    if x[-1]>3:
        my_xticks = np.arange(0, x[-1], 1)
    elif (x[-1]<3)&(x[-1]>1.5):
        my_xticks = np.arange(0, x[-1], 0.5)
    elif x[-1]<1.5:
        my_xticks = np.arange(0, x[-1], 0.3)
    return my_xticks        

def plot_EMG_spectrogram(data, EMG_names, fs, plt_start_time, plt_end_time, f_range = [0, 400]):
    """
    This function is used to calculate and plot the spectrogram of multi-ch EMG signals.
    It calls spectrogram in scipy.signal to do the computation
    
    data: the EMG signals you want to analyze, a T*n numpy array, T is the number of
          samples, n is the number of channels
    EMG_names: a list for the names of EMG channels or labels for forces
    fs: the sampling frequency
    plt_start_time: the start time for plotting, a float number
    plt_end_time: the end time for plotting, a float number
    f_range: a two-element list specifying the start and the end frequency you want to plot,
            default is from 0 Hz to 400 Hz
    """
    N = data.shape[1]
    grid = plt.GridSpec(N, 1, wspace=0.5,hspace=0.2)
    cmap = plt.cm.jet
    for i in range(N):
        ax = plt.subplot(grid[i,0])
        ax.set_ylabel('f (Hz)', fontsize = 18)
        ax.tick_params(axis=u'both', which=u'both',length=4)
        plt.tick_params(labelsize=18)
        sns.despine()
        f, t, Sxx = signal.spectrogram(data[int(plt_start_time*fs):int(plt_end_time*fs), i], fs, 
                               scaling = 'density', nperseg = 256, noverlap = 64, nfft = 256)
        f_idx = np.where((f>f_range[0])&(f<f_range[1]))[0]
        plt.text(1, f[f_idx[-1]] ,'%s' %(EMG_names[i]),fontsize = 18,
                 verticalalignment="top",horizontalalignment="left")
        if i<N-1:
            im = ax.pcolormesh(t, f[f_idx], 10*np.log10(Sxx[f_idx, :]), cmap = cmap)
            plt.colorbar(im, ax = ax)
            plt.setp(ax.get_xticklabels(),visible=False)    
        if i == N-1:
            im = ax.pcolormesh(t, f[f_idx], 10*np.log10(Sxx[f_idx, :]), cmap = cmap)
            plt.colorbar(im, ax = ax)
            plt.setp(ax.get_xticklabels(),visible=True)
            ax.set_xlabel('Time (s)', fontsize = 18)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize = 14)
        cbar.set_label('dB', fontsize = 14)

def wireless_dropout_detection(behave_seg, threshold):
    """
    To determine whether there is any dropout within a specific behavior segmentation.
    behave_seg: the behavior segmentation dictionary list corresponding to a 15-min file
    threshold: the threshold to determine whether a segment should be abadoned
    """
    idx = []
    for k, each in enumerate(behave_seg):
        seg_30k = each['30k']
        diff_seg_30k = np.array([np.diff(seg_30k[:, i]) for i in range(3)]).T
        a = []
        for i in range(3):
            a.append(max([len(list(v)) for k, v in itertools.groupby(diff_seg_30k[:, i])]))
        if max(a) > threshold:
            idx.append(k)
    idx_ = list(np.flip(sorted(idx)))
    for each in idx_:
        behave_seg.pop(each)
    return behave_seg
        
def plot_behave_dict_spike_timing_raw_EMG(behave_dict, raw_flag = 1, bin_size = 0, offset = 0, EMG_chs = 0):
    if raw_flag == 1:
        bin_size = 1/behave_dict['raw_EMG_fs']
    else:
        bin_size = bin_size       
    print(behave_dict['label'])
    if EMG_chs == 0:
        EMG_chs = np.arange(behave_dict['raw_EMG'].shape[1])        
    p_names = behave_dict['EMG_names']
    N = len(EMG_chs)
    spike_grid = 5
    p_spike, p_emg = behave_dict['spikes'], behave_dict['raw_EMG']
    grid = plt.GridSpec(N+spike_grid,1,wspace=0.5,hspace=0.2)
    main_ax = plt.subplot(grid[0:spike_grid,0])
    for i, spiketrain in enumerate(behave_dict['spike_timing']):
        main_ax.plot(spiketrain - offset, np.ones_like(spiketrain) * i, ls='', marker='|', color = 'k', ms = 1)

    x = np.arange(behave_dict['raw_EMG'].shape[0])*bin_size - offset
    if 'pg' in behave_dict['label']:
        fsr_ax = main_ax.twinx()
        t_fsr = behave_dict['timeframe'] - behave_dict['timeframe'][0] - offset
        fsr_ax.plot(t_fsr, behave_dict['FSR_data'][:, 0], 'blue')
        fsr_ax.plot(t_fsr, behave_dict['FSR_data'][:, 1], 'royalblue')
        fsr_ax.axis('off')
    main_ax.axis('off')
    plt.xticks(color = 'w')
    plt.yticks([])
    #ylim_num = 1*np.max(p_emg[plot_start:plot_start+plot_len, :])
    for i in range(N):
        ax0 = plt.subplot(grid[i+spike_grid,0], sharex = main_ax)
        p1 = p_emg[:, EMG_chs[i]]
        #plt.yticks([])
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        if i<N-1:
            plt.plot(x, p1, 'k')
            ax0.spines['bottom'].set_visible(False)
            plt.setp(ax0.get_xticklabels(),visible=False)
            ax0.tick_params(axis=u'both', which=u'both',length=0)
        if i == N-1:
            ax0.tick_params(axis=u'both', which=u'both',length=4)
            plt.setp(ax0.get_xticklabels(),visible=True)
            plt.plot(x, p1, 'k')
            #plt.xticks(get_time_ticks(x), color='k')
            ax0.set_xlabel('Time (s)', fontsize = 20)
            plt.tick_params(labelsize = 16)
            labels = ax0.get_xticklabels() + ax0.get_yticklabels()
            [label.set_fontname('Arial') for label in labels]
            #ax0.set_xticks(np.arange(0, len(p1), 500))           
        #plt.ylim(0, 200)
        
        plt.text(x[-1], np.max(p1),'%s' %(p_names[EMG_chs[i]]),fontsize = 16, color = 'k',
                  verticalalignment="top",horizontalalignment="left")    
    
def pull_spike_EMG_from_behave_dict(behave_dict, EMG_idx = []):
    if EMG_idx == []:
        EMG_idx = np.arange(behave_dict[0]['EMG'].shape[1])
    spikes = [each['spikes'] for each in behave_dict]
    EMG = [each['EMG'][:, EMG_idx] for each in behave_dict]
    if 'pg' in behave_dict[0]['label']:
        return spikes, EMG, [each['FSR_data'] for each in behave_dict]
    else:
        return spikes, EMG

def find_non_drop_out(x, y, TR, TH = 1, t = []):
    """
    A function to get the non-drop-out portion of the data
    x is the signal with potential dropouts, like spikes in the cage
    y is the signal to be decoded, like the EMGs
    Both x and y are numpy arrays with the shape Time x D
    TR is a threshold to discard too short periods
    TH is a threshold to avoid the data right before or after drop-outs
    The outputs were lists.
    t is optional. If t is empty, the function will skip it. If not, the function will find out the time stamps corresponding to non-drop-out segments
    """
    if isinstance(x, list) == True:
        x = np.array(x).T
        y = np.array(y).T
    
    s = np.sum(x, axis = 1)
    
    z = np.where(s == 0)[0]
    idx_dz = np.where(np.diff(z)>1)[0]

    idx_dz_ = idx_dz + 1
    idx_z = [z[each] for each in idx_dz]
    idx_z.append(z[-1])

    idx_z_ = [z[each] for each in idx_dz_]
    idx_z_.insert(0, z[0])

    idx = list(np.array(idx_z) + TH)
    idx_ = list(np.array(idx_z_) - TH)

    idx.insert(0, 0)
    idx_.append(len(s))
    
    x_, y_, t_ = [], [], []
    for each in zip(idx, idx_):
        x_.append(x[each[0]:each[1], :])
        y_.append(y[each[0]:each[1], :])
        if len(t)>0:
            t_.append(t[each[0]:each[1]])
    L = np.array([len(each) for each in x_])
    I = list(np.where(L<TR)[0])
    I.sort(reverse=True)
    for each in I:
        x_.pop(each)
        y_.pop(each)
        if len(t_)>0:
            t_.pop(each)
    if len(t_)>0:
        return x_, y_, t_
    else:
        return x_, y_    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


