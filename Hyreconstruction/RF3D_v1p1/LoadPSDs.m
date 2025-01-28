function [ psd_rnd, psd_fpn ] = LoadPSDs( transform_2D_name )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  LoadPSDs loads the 8x8 PSDs of the random and fixed-pattern noise defined
%  for FLIR Tau 320 with respect to the specified 2D transform.
%  The PSDs are loaded from the
%  provided mat files.
%
% [1] M. Maggioni, E. Sanchez-Monge, A. Foi, "Joint removal of random and
%     fixed-pattern noise through spatiotemporal video filtering", IEEE 
%     Transactions on Image Processing, vol.23, no.10, pp. 4282-4296, Oct. 2014
%     http://doi.org/10.1109/TIP.2014.2345261
% 
%  FUNCTION INTERFACE:
%
%  [ psd_rnd, psd_fpn ] = LoadPSDs( transform_2D_name )
%
%  INPUTS
%     1) transform_2D_name   (char) : 2D spatial transform defining the
%                                     PSDs. We support 'dct', 'bior1.5',
%                                     and 'haar' transform. 
% 
%  OUTPUTS:
%     1) psd_rnd         (2D array) : PSD of the random noise.
%     2) psd_fpn         (2D array) : PSD of the fixed-pattern noise.
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


if strcmpi(transform_2D_name,'dct')
    load PSDs_dct.mat
elseif strcmpi(transform_2D_name,'bior1.5')
    load PSDs_bior.mat
elseif strcmpi(transform_2D_name,'haar')
    load PSDs_haar.mat
else
    warning('PSDs for "%s" transform not provided, using DCT instead.', upper(transform_2D_name))
    warning('The %s transform will be used in RF3D, but is not consistent with the DCT transform defining the PSDs.', upper(transform_2D_name))
    load PSDs_dct.mat
end

N = 8; % must not be changed
psd_rnd = PSDs.RND{N};
psd_fpn = PSDs.FPN{N};
if isempty(psd_rnd)
    error('PSD of random noise does not exist for given size %dx%d.', N,N)
elseif isempty(psd_fpn)
    error('PSD of fixed-pattern noise does not exist for given size %dx%d.', N,N)
end
% normalize PSD matrices with respect to their highest-frequency term
psd_rnd=psd_rnd/max(eps,psd_rnd(end));
psd_fpn=psd_fpn/max(eps,psd_fpn(end));

end

