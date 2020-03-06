function NEV = save_nev_as_mat(base_path, file_name, nsx)
if nargin < 3
    nsx = '.ns3';
end
has_NSx = 0;
raw_NEV = openNEV(strcat(base_path, file_name), 'nosave', 'nomat');
if exist(strcat(base_path, strcat(file_name(1:end-4),nsx)),'file')
   raw_NSx = openNSx(strcat(base_path, strcat(file_name(1:end-4),nsx)));
   has_NSx = 1;
end

NEV.date_time = raw_NEV.MetaTags.DateTime;
NEV.fs = raw_NEV.MetaTags.SampleRes;
NEV.duration = raw_NEV.MetaTags.DataDurationSec;
NEV.comment = raw_NEV.MetaTags.Comment;

elec_labels_raw = transpose([raw_NEV.ElectrodesInfo.ElectrodeLabel]);
elec_labels = string(elec_labels_raw);
idx = find(cell2mat(strfind(elec_labels, 'elec')) == 1);
NEV.elec_labels = elec_labels_raw(idx, 1:8);
elec_id = transpose([raw_NEV.ElectrodesInfo.ElectrodeID]);
NEV.elec_id = elec_id(idx);
elec_bank = transpose([raw_NEV.ElectrodesInfo.ConnectorBank]);
NEV.elec_bank = elec_bank(idx);
elec_pin = transpose([raw_NEV.ElectrodesInfo.ConnectorPin]);
NEV.elec_pin = elec_pin(idx);
elec_threshold = transpose([raw_NEV.ElectrodesInfo.LowThreshold]);
NEV.elec_threshold = elec_threshold(idx);
elec_units = transpose([raw_NEV.ElectrodesInfo.Units]);
NEV.elec_units = elec_units(idx);

NEV.data.serial = raw_NEV.Data.SerialDigitalIO;
NEV.data.spikes = raw_NEV.Data.Spikes;
NEV.data.comments = raw_NEV.Data.Comments;

if has_NSx == 1
    NEV.analog_fs = raw_NSx.MetaTags.SamplingFreq;
    NEV.analog_MinDigiValue = raw_NSx.ElectrodesInfo.MinDigiValue;  
    NEV.analog_MaxDigiValue = raw_NSx.ElectrodesInfo.MaxDigiValue;
    NEV.analog_MinAnalogValue = raw_NSx.ElectrodesInfo.MinAnalogValue;
    NEV.analog_MaxAnalogValue = raw_NSx.ElectrodesInfo.MaxAnalogValue;
    NEV.analog_labels = [];
    for i = 1:length(raw_NSx.ElectrodesInfo)
        NEV.analog_labels = [NEV.analog_labels; raw_NSx.ElectrodesInfo(i).Label];
    end
    NEV.analog_data = raw_NSx.Data;
end

end

