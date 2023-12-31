function [C,KSq] = getDk(Params, R, datatype,n_fft, kerneltype,divide)
% [C, KSq] = getDk(Params, R, datatype,n_fft, kerneltype,divide)
% Author: Xu Li
% Affiliation: Radiology @ JHU
% Email address: xuli@mri.jhu.edu
% Updated X.L., 2019-07


if nargin < 3
    datatype = 'single';
    n_fft=1;
    kerneltype  = 0;
    divide=3;
elseif nargin < 4
    n_fft=1;
    kerneltype = 0;   
    divide=3;
elseif nargin < 5
    kerneltype = 0;   
    divide=3;
elseif nargin < 6
    divide=3;
end

if isempty(datatype)
    datatype = 'single';
end

if ~isfield(Params, 'permuteflag')
    Params.permuteflag = 1;     % version before v2.8
end
    
warning off all
  
Nx = Params.sizeVol(1).*n_fft;
Ny = Params.sizeVol(2).*n_fft;
Nz = Params.sizeVol(3).*n_fft;

dkx = 1/Params.fov(1);  % 
dky = 1/Params.fov(2);
dkz = 1/Params.fov(3);

%% convolution kernel 
Nx=single(Nx);
Ny=single(Ny);
Nz=single(Nz);
kx = linspace(-Nx/2, Nx/2-1, Nx).*dkx;   % size * 1/FOV= pixel per mm    
ky = linspace(-Ny/2, Ny/2-1, Ny).*dky;
kz = linspace(-Nz/2, Nz/2-1, Nz).*dkz;
if strcmp(datatype, 'single')
    kx = single(kx);
    ky = single(ky);
    kz = single(kz);
end

[KX_Grid, KY_Grid, KZ_Grid] = meshgrid(ky, kx, kz);
% disp(size(KX_Grid))
% disp(size(KY_Grid))
KSq = KX_Grid.^2 + KY_Grid.^2 + KZ_Grid.^2;  

if isfield(Params, 'Tsominv') && isfield(Params, 'Tpom')  % #       
    
    H0 = [0, 0, 1]';                   % Lab space XYZ  
    extra = [0,-1,0;-1,0,0;0,0,1];     %     
    Hsub = -extra*Params.Tsominv*R'*Params.Tpom*H0;                    
    R31 = Hsub(1);
    R32 = Hsub(2);
    R33 = Hsub(3);
        
elseif isfield(Params, 'sliceOri')
    
    switch Params.sliceOri
        case 1  % axial
            H0 = [0, 0, 1]';    % Lab space XYZ
        case 2
            H0 = [1, 0, 0]';    % need further testing
        case 3  % coronal 
            H0 = [0, 1, 0]';    % in Y directoin
        otherwise
            error('unknown slice orientation.')
    end
    
    Hsub = R'*H0;              
    R31 = Hsub(1);
    R32 = Hsub(2);
    R33 = Hsub(3);
    
else
    % default use axial H0 = [0, 0, 1], Hsub = R'*H0    
    R31 = R(3, 1);
    R32 = R(3, 2);
    R33 = R(3, 3);
end

KZp_Grid = R31 .* KX_Grid + R32 .* KY_Grid + R33.* KZ_Grid;  %  i.e. ((R'*H0).*K), in subject frame
   
clear KX_Grid KY_Grid KZ_Grid

if kerneltype == 0              
    C = 1/divide - (KZp_Grid.^2)./(KSq);         % normal D2 kernel
    clear KZp_Grid 
    C(isnan(C)) = 0;  

elseif kerneltype == 1    
    C = 1/divide.*KSq - KZp_Grid.^2;             % Laplacian DL kernel
    
end



