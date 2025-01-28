function [ z, psd_rnd, psd_fpn ] = LoadRealDataFLIR( file_name, transform_2D_name )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  LoadRealData loads a sequence acquired by a FLIR Tau 320 camera and the
%  corresponding PSDs of the random and fixed-pattern noise.
%
% [1] M. Maggioni, E. Sanchez-Monge, A. Foi, "Joint removal of random and
%     fixed-pattern noise through spatiotemporal video filtering", IEEE 
%     Transactions on Image Processing, vol.23, no.10, pp. 4282-4296, Oct. 2014
%     http://doi.org/10.1109/TIP.2014.2345261
% 
%  FUNCTION INTERFACE:
%
%  [ z, psd_rnd, psd_fpn ] = LoadRealDataFLIR( file_name, transform_2D_name )
%
%  INPUTS
%     1) file_name           (char) : Path to the local file (optional). If
%                                     empty, a pompt dialog will appear
%                                     letting the user select a file.
%     2) transform_2D_name   (char) : 2D spatial transform defining the
%                                     PSDs. We support 'dct', 'bior1.5',
%                                     and 'haar' transform. 
% 
%  OUTPUTS:
%     1) z               (3D array) : Matlab array containing the video.
%     2) psd_rnd         (2D array) : PSD of the random noise.
%     3) psd_fpn         (2D array) : PSD of the fixed-pattern noise.
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
if ~exist('transform_2D_name','var')
    error('Not enough input arguments.')
end

% If no file_name is given, let the user select a file from local disk
if ~exist('file_name','var') || isempty(file_name)
    [ file_name, folder ] = uigetfile({'*.tif;*.tiff'}, 'Select FLIR Tau 320 sequence');
    if isequal(file_name,0)
        error('No file selected.')
    end
    file_name = [folder,file_name];
end

% Quit if the file is not supported
if ~(strcmp(file_name(end-3:end),'.tif') || strcmp(file_name(end-3:end),'tiff'))
    error('File type "%s" not supported. Please select a TIF file.', file_name(end-3:end))
end

% Loading sequence
fprintf('Loading FLIR Tau 320 sequence "%s" \n', file_name);
info = imfinfo(file_name);
nFrames = numel(info);
z = zeros(info(1).Height,info(1).Width,nFrames, 'single');
for k = 1:nFrames
    A = imread(file_name, k, 'Info', info);
    z(:,:,k) = single(A);
end

% Loading PSDs
[ psd_rnd, psd_fpn ] = LoadPSDs( transform_2D_name );

end