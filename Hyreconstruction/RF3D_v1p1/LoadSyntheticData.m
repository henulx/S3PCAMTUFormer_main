function [ y, z, psd_rnd, psd_fpn, FPN ] = LoadSyntheticData( file_name, transform_2D_name, sigma_rnd, sigma_fpn, FPN_drift, seed )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  LoadSyntheticData loads a noise-free sequence and add synthetic noise of
%  given parameters. The function also returns the PSDs of the random and
%  fixed-pattern noise for the assumed noise model.
%
% [1] M. Maggioni, E. Sanchez-Monge, A. Foi, "Joint removal of random and
%     fixed-pattern noise through spatiotemporal video filtering", IEEE
%     Transactions on Image Processing, vol.23, no.10, pp. 4282-4296, Oct. 2014
%     http://doi.org/10.1109/TIP.2014.2345261
%
%  FUNCTION INTERFACE:
%
%  [ y, z, psd_rnd, psd_fpn ] = LoadSyntheticData( file_name, transform_2D_name, sigma_rnd, sigma_fpn, FPN_drift, seed )
%
%  INPUTS
%     1) file_name           (char) : Path to the local file (optional). If
%                                     empty, a pompt dialog will appear
%                                     letting the user select a file.
%     2) transform_2D_name   (char) : 2D spatial transform defining the
%                                     PSDs. We support 'dct', 'bior1.5',
%                                     and 'haar' transform.
%     3) sigma_rnd         (double) : Standard deviation of the random
%                                     noise component (RND).
%     4) sigma_fpn         (double) : Std. dev. of the fixed-pattern noise
%                                     component.
%     5) FPN_drift         (double) : Drift parameter of the FPN component.
%                                     Use 0 for static FPN.
%                                     Use 0<lambda_FPNdrift<=1 for a
%                                     drifting FPN component.
%                                     (default is 0).
%     6) seed              (double) : Seed for the random number generator.
%                                     (default is 0).
%
%  OUTPUTS:
%     1) y               (3D array) : Matlab array containing the noise-free video.
%     2) z               (3D array) : Matlab array containing the noisy video.
%     3) psd_rnd         (2D array) : PSD of the random noise.
%     4) psd_fpn         (2D array) : PSD of the fixed-pattern noise.
%     5) FPN             (3D array) : Fixed-pattern noise corrupting the
%                                     video
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2011-2020    All rights reserved.
% This work should only be used for nonprofit purposes.
%
% AUTHORS:
%     Enrique Sanchez-Monge  < esm _at_ noiselessimaging.com >
%     Matteo Maggioni 
%     Alessandro Foi  < alessandro.foi _at_ tuni.fi >
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check input arguments
if ~exist('transform_2D_name','var') || ~exist('sigma_rnd','var') || ~exist('sigma_fpn','var')
    error('Not enough input arguments.')
end
if ~exist('FPN_drift','var')
    FPN_drift = 0;
else
    FPN_drift = max(min(FPN_drift,1),0);
end
if ~exist('seed','var')
    seed = 0;
end

% If no file_name is given, let the user select a file from local disk
if ~exist('file_name','var') || isempty(file_name)
    [ file_name, folder ] = uigetfile({'*.avi'}, 'Select noise-free video file');
    if isequal(file_name,0)
        error('No file selected.')
    end
    file_name = [folder,file_name];
end

% Quit if the file is not supported
if ~strcmp(file_name(end-3:end),'.avi')
    error('File type "%s" not supported. Please select an AVI file.', file_name(end-3:end))
end

% Loading sequence
fprintf('Loading test sequence "%s" \n', file_name);
mov       = VideoReader(file_name);
nFrames   = mov.NumberOfFrames;
vidHeight = mov.Height;
vidWidth  = mov.Width;
y = zeros(vidHeight,vidWidth,nFrames, 'single');
for n = 1:nFrames
    frame     = single((read(mov, n)));
    if size(frame,3) == 1
        y(:,:,n)  = single(frame);
    else
        y(:,:,n)  = single(rgb2gray(frame));
    end
end

% Setting seed
try
    randn('seed',seed); % using this will yield the results in [1]
catch
    rng(seed)
end


% Loading global PSDs for noise generation
if exist('PSDkernels.mat','file')
    load PSDkernels.mat PSDkernelRND PSDkernelFPN
else
    error('File "PSDkernels.mat" (mat file containing PSD kernels of random and fixed-pattern noise) does not exist.')
end

% Loading PSDs [NOTE: these PSDs and the global PSDs need to be matched with each other.]
[ psd_rnd, psd_fpn ] = LoadPSDs( transform_2D_name );


% Adding noise
if FPN_drift>0
    fprintf('Adding synthetic noise with sigma_rnd=%.1f and sigma_fpn=%.1f (FPN_drift=%.2f) \n', sigma_rnd, sigma_fpn, FPN_drift);
else
    fprintf('Adding synthetic noise with sigma_rnd=%.1f and sigma_fpn=%.1f (no drift)\n', sigma_rnd, sigma_fpn);
end
size_z_1 = size(y,1);
size_z_2 = size(y,2);
z   = zeros(size(y));
FPN = zeros(size(y));
for jj=1:size(y,3)
    rnd=ifft2(fft2(sigma_rnd*randn(size_z_1,size_z_2)).*fft2(PSDkernelRND,size_z_1,size_z_2));
    fpn=ifft2(fft2(sigma_fpn*randn(size_z_1,size_z_2)).*fft2(PSDkernelFPN,size_z_1,size_z_2));
    if ~exist('fpn_old','var')
        fpn_old=fpn;
        lambda=1;
    else
        lambda=FPN_drift;
    end
    fpn=fpn*lambda^0.5+fpn_old*(1-lambda)^0.5;
    z(:,:,jj)   = y(:,:,jj)+fpn+rnd;
    FPN(:,:,jj) = fpn;
    fpn_old=fpn;
end


end