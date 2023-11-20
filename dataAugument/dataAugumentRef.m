% Author: Joe
% Affiliation: XMU
% Create: 2022-12
% Updated Joe: 2023-11

%% linear augument
len_path=length(data_path_list);
for i = 1:len_path
% Load data 
data_process = char(data_path_list(i));  
fprintf('Loading: %s \n',data_process);  
load(data_process); 

size_ori_data=size(chi_mo); 
datatype = 'single';
[D_k,~] = getDk(Params, Params.TAng, datatype,1,0,3);
k=2*rand(1)+1;
chi_mo = k*chi_mo;
chi_data_fft = fftshift(fftn(chi_mo));
B_k=D_k.*chi_data_fft;             % get deltaB in k space            
randon_complex_data =  1*randn(size_ori_data)+1i*1*randn(size_ori_data);
deltaB = real(ifftn(ifftshift(B_k+randon_complex_data)));  % add random noise
% save data
save_name= [save_path,'chi_add',data_name,'.mat'];

end


%% add chi source
len_path=length(data_path_list);
for i = 1:len_path
% Load data 
data_process = char(data_path_list(i));  
fprintf('Loading: %s \n',data_process);  
load(data_process); 

% add chi source
n_time = 100; % the time to add chi source
[~,Mag,chi_mo_temp] = getDataAugumentV2(chi_mo,maskErode,Params,n_time);

size_ori_data=size(chi_mo_temp); 
datatype = 'single';
[D_k_temp,~] = getDk(Params, Params.TAng, datatype,1,0,3);
chi_data_fft = fftshift(fftn(chi_mo_temp));
B_k=D_k_temp.*chi_data_fft;        
randon_complex_data = 0.5*randn(size_ori_data)+1i*0.5*randn(size_ori_data);
deltaB = real(ifftn(ifftshift(B_k+randon_complex_data)));

save_name= [save_path,'chi_add',data_name,'.mat'];
% save data 
chi_mo = chi_mo_temp;

end



