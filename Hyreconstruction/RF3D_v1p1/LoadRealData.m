function [ z ] = LoadRealData( file_name )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  LoadRealData loads a sequence. Modify this function to your own need.
% 
%  FUNCTION INTERFACE:
%
%  [ z ] = LoadRealData( file_name )
%
%  INPUTS
%     1) file_name           (char) : Path to the local file (optional). If
%                                     empty, a pompt dialog will appear
%                                     letting the user select a file.
%  OUTPUTS:
%     1) z               (3D array) : Matlab array containing the video.
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


% If no file_name is given, let the user select a file from local disk
if ~exist('file_name','var') || isempty(file_name)
    [ file_name, folder ] = uigetfile({'*.*'}, 'Select  sequence');
    if isequal(file_name,0)
        error('No file selected.')
    end
    file_name = [folder,file_name];
end

% Get extension
[~,~,ext] = fileparts(file_name);

if strcmp(ext,'.tiff') || strcmp(ext,'.tif')
    % Loading Tiff sequence
    fprintf('Loading TIFF sequence "%s" \n', file_name);
    info = imfinfo(file_name);
    nFrames = numel(info);
    z = zeros(info(1).Height,info(1).Width,nFrames, 'single');
    for k = 1:nFrames
        A = imread(file_name, k, 'Info', info);
        z(:,:,k) = single(A);
    end
else  
    fprintf('Loading test sequence "%s" \n', file_name);
    mov       = VideoReader(file_name);
    nFrames   = mov.NumberOfFrames;
    vidHeight = mov.Height;
    vidWidth  = mov.Width;
    z = zeros(vidHeight,vidWidth,nFrames, 'single');
    for n = 1:nFrames
        frame     = single((read(mov, n)));
        if size(frame,3) == 1
            z(:,:,n)  = single(frame);
        else
            z(:,:,n)  = single(rgb2gray(frame));
        end
    end
end
end