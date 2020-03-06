base_dir = 'D:\data\cage_data\samples_for_cage_denoising\';
open_file = strcat(base_dir, '*.nev');
file = dir(open_file);
good_waves = struct([]);
bad_waves = struct([]);
for i = 1:length(file)
    file_name = file(i).name;
    raw_NEV = openNEV(strcat(base_dir, file_name), 'nosave', 'nomat');
    bad_idx = find(raw_NEV.Data.Spikes.Unit == 255);
    good_idx = find(raw_NEV.Data.Spikes.Unit == 1);
    good_waves{i, 1} = transpose(raw_NEV.Data.Spikes.Waveform(:, good_idx));
    bad_waves{i, 1} = transpose(raw_NEV.Data.Spikes.Waveform(:, bad_idx));
end
save_file = strcat(base_dir, 'Greyson_waveforms.mat');
save(save_file, 'good_waves', 'bad_waves');









