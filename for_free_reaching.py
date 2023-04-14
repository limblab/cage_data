import h5py
import time
import numpy as np
from scipy import stats, signal

   
def parse_nev_mat_file(filename, has_analog = 0, empty_channels = []):
    """
    Parse MATLAB version of .nev files
    """
    py_nev = {}
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
    waveform = np.asarray(nev_file['data']['spikes']['Waveform'])
    s_spikes = []
    s_waveforms = []
    for each in elec_id:
        idx = np.where(electrode == each)[0]
        s_spikes.append(time_stamp[idx])
        s_waveforms.append(waveform[idx,:])

    py_nev['nev_fs'] = nev_file['fs'][0][0]
    py_nev['nev_duration'] = nev_file['duration'][0][0]
    py_nev['electrode_meta'] = electrode_meta
    py_nev['thresholds'] = thresholds
    py_nev['spikes'] = s_spikes
    py_nev['waveforms'] = s_waveforms
    py_nev['ch_lbl'] = ch_lbl
    
    if has_analog:
       py_nev['analog_fs'] = nev_file['analog_fs'][0][0]
       py_nev['analog_data'] = np.asarray(nev_file['analog_data'])
       analog_lbl = list(np.asarray(nev_file['analog_labels']).T)
       for (i, each) in enumerate(analog_lbl):
           analog_lbl[i] = ''.join(chr(x) for x in each).strip(b'\x00'.decode())
       py_nev['analog_lbl'] = analog_lbl
       py_nev['analog_MaxDigiValue'] = nev_file['analog_MaxDigiValue'][0][0]
       py_nev['analog_MaxAnalogValue'] = nev_file['analog_MaxAnalogValue'][0][0]
       py_nev['analog_MinDigiValue'] = -py_nev['analog_MaxDigiValue']
       py_nev['analog_MinAnalogValue'] = -py_nev['analog_MaxAnalogValue']
    
    e = time.clock()
    print("%.3f s for parsing the nev-mat file" %(e-s))
    return py_nev

def bin_spikes(spikes, file_length, bin_size, mode = 'center'):
    """
    Take a list containing spike times as input, return binned firing rates
    """
    print('Binning spikes with %.4f s' % (bin_size))
    binned_spikes = []
    if mode == 'center':
        bins = np.arange(bin_size - bin_size/2, 
                          file_length + bin_size/2, bin_size)
    elif mode == 'left':
        bins = np.arange(0, file_length, bin_size)
    bins = bins.reshape((len(bins),))
    for each in spikes:
        each = each/30000
        each = each.reshape((len(each),))
        out, _ = np.histogram(each, bins)
        binned_spikes.append(out)
    bins = np.arange(0, file_length, bin_size)
    return bins[1:], binned_spikes            

def smooth_binned_spikes(binned_spikes, kernel_type, bin_size, kernel_SD, sqrt = 0):
    """
    Binned spikes are stored in a list, sqrt specifies whether to perform square root transform
    """
    smoothed = []
    if sqrt == 1:
       for (i, each) in enumerate(binned_spikes):
           binned_spikes[i] = np.sqrt(each)
    kernel_hl = np.ceil( 3 * kernel_SD / bin_size )
    normalDistribution = stats.norm(0, kernel_SD)
    x = np.arange(-kernel_hl*bin_size, (kernel_hl+1)*bin_size, bin_size)
    kernel = normalDistribution.pdf(x)
    if kernel_type == 'gaussian':
        pass
    elif kernel_type == 'half_gaussian':
       for i in range(0, int(kernel_hl)):
            kernel[i] = 0
    n_sample = np.size(binned_spikes[0])
    nm = np.convolve(kernel, np.ones((n_sample))).T[int(kernel_hl):n_sample + int(kernel_hl)] 
    for each in binned_spikes:
        temp1 = np.convolve(kernel,each)
        temp2 = temp1[int(kernel_hl):n_sample + int(kernel_hl)]/nm
        smoothed.append(temp2)
    print('The binned spikes have been smoothed.')
    return smoothed
  
if __name__ == '__main__':
    data_path = 'D:/data/lab_data/20191101/'
    # ---------- The first file is a MATLAB version of .nev file ---------- #
    nev_file = '20191101_Greyson_freereaching_001.mat'
    filename = ''.join((data_path, nev_file))
    # ---------- Parsing the data file ---------- #
    py_nev = parse_nev_mat_file(filename, 1)
    # ---------- Binning the spikes with 0.05 s bins ---------- #
    time_frame, binned_spikes = bin_spikes(py_nev['spikes'], py_nev['nev_duration'], 0.05)
    # ---------- Smoothing the binned spikes using Gaussian kernel with SD = 0.1 ---------- #
    smoothed_binned_spikes = smooth_binned_spikes(binned_spikes, 'gaussian', 0.05, 0.1)
    # ---------- How to get video sync pulses ---------- #
    video_sync_pulses = py_nev['analog_data'][:, 12]
    
    
    
    