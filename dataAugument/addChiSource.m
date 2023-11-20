%% 
function [deltaB,mag,output_chi] = addChiSource(chi_mo,maskErode,Params,n_time,th,filter_flag)
% [deltaB,mag,output_chi] = addChiSource(chi_mo,Mag,maskErode,Params,n_time,th,filter_flag)
% Author: Joe
% Affiliation: XMU
% Create: 2022-12
% Updated Joe: 2023-11

if nargin<5
    n_time = 100; % the numeber the chi source 
    th =0.15;     % threshold for chi to get the Mag
    filter_flag=1;
end

if nargin<6
    th = 0.2;     % threshold for chi to get the Mag
end

if nargin <7 
   filter_flag=1; 
end
% preprocessing 
chi_temp = chi_mo; 
chi_temp = 1.1*chi_temp;

result_out = chi_temp;
size_data = size(chi_temp);
delta=20;           
kernel_type_list = ["boll";"ellipse";"line_x";"line_y";"line_z";"line_rand";"line_circle"];
value_list = [-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4];
try_time_max = 10; 
range = 3; 
width = [2,3,4]; 
output_temp = result_out; 
d = 6;     %  distance between chi source
for time=1:n_time   
    try_time = 1;   
    mode_select = randi(2);
%   mode_select = 7;
%   select the location 
    locationx = delta+randi(size_data(1)-2*delta);
    locationy = delta+randi(size_data(2)-2*delta);
    locationz = delta+randi(size_data(3)-2*delta);
    
%   fprintf('locationx,locationy,locationz :%d,%d,%d\n',locationx,locationy,locationz);
    abs_temp = abs(output_temp(locationx-d:locationx+d,locationy-d:locationy+d,locationz-d:locationz+d));
    mask_temp = maskErode(locationx-6:locationx+6,locationy-6:locationy+6,locationz-2:locationz+2);
    while max(abs_temp(:))>0.10 && try_time <try_time_max || max(mask_temp(:))-min(mask_temp(:))==1
        locationx = delta+randi(size_data(1)-2*delta);
        locationy = delta+randi(size_data(2)-2*delta);
        locationz = delta+randi(size_data(3)-2*delta);
        try_time = try_time+1; 
        abs_temp = abs(output_temp(locationx-d:locationx+d,locationy-d:locationy+d,locationz-d:locationz+d));
        mask_temp = maskErode(locationx-3:locationx+3,locationy-3:locationy+3,locationz-2:locationz+2);
    end
    value_temp =  value_list(randi(8)); 
%   fprintf('try_time: %d\n',try_time)  
    kernel_type = char(kernel_type_list(max(mode_select,1)));  
%   fprintf('* No:%d,model_select: %d,kernel_type: %s, value_temp: %.2f \n',time,mode_select,kernel_type,value_temp)    
    if mode_select <= 1 
        range = 2+randi(2); % boll radius 
    elseif mode_select ==2  
        width = [1+randi(3),1+randi(3),1+randi(3)];
        range = 1+1*rand(1);  % the basic ratio for a,b,c  
    end 
    location = [locationx,locationy,locationz];
    output_temp = set_kernel_data(output_temp,location,range,value_temp,kernel_type,width);
    
end  
% mask_Erode1 = imerode3(maskErode,3,1); 
% add noise      
output_chi = (output_temp+0.01*rand(size(maskErode))).*maskErode; % the final chi source
disp('Done !')
% get mag file   
mask = single(abs(chi_mo)>th);

 
% get deltab    
datatype='single';     
[D_k_temp,DK_mask,Ksq] = D_k_define(Params, Params.TAng, datatype,1,0,3,0.1); 

diff_mask = single(abs(output_chi-chi_mo)>0.05);
mag = get_mag_val(Mag,chi_mo,diff_mask); 
if filter_flag
    chi_filter = gauss_filter(output_chi);
    diff_mask_dilate = imdilate(diff_mask, strel('disk', 3));
    output_chi(diff_mask_dilate==1) = 0.5*chi_filter(diff_mask_dilate==1)+0.5*chi_mo(diff_mask_dilate==1);
    output_chi(diff_mask==1) = chi_filter(diff_mask==1);
end

chi_data_fft=fftshift(fftn(output_chi));    % get fft data     
B_k=D_k_temp.*chi_data_fft;             % get deltaB in k space   
deltaB=real(ifftn(ifftshift(B_k))).*maskErode;   % get deltaB in image space  
output_chi = output_chi.*maskErode;

function value_get = get_mag_val(mag,chi_mo,mask)
% get mag 
val = mag;
mag_value = mag(abs(chi_mo)>0.09);
mag_mean = mean(mag_value(:));
temp = 0.2*abs((abs(chi_mo)-0.1));

mag_temp = abs(mag_mean-temp);
val(mask>0)=mag_temp(mask>0);
value_get = val;
end

end


