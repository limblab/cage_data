clc
clear
base_dir = 'D:\data\cage_data\20190304\';
save_dir = base_dir; 
open_file = strcat(base_dir, '*.nev');
file = dir(open_file);
for i = 1:length(file)
    file_name_in_list = file(i).name;
    disp(file_name_in_list);
    NEV = save_nev_as_mat(base_dir, file_name_in_list);
    save_file = strcat(file_name_in_list(1:end-4), '.mat');
    save(strcat(save_dir, save_file), 'NEV', '-v7.3');
    %clear NEV
end