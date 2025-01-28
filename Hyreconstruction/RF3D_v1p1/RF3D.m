function y_est = RF3D(z, sigma_rnd, sigma_fpn, psd_rnd, psd_fpn, transform_2D_name, erf3d, do_wiener, profile, print_to_screen)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  RF3D (ver. 1.1, 7 April 2020)
%
%  RF3D is an algorithm for joint attenuation of random (RND) and Fixed-
%  Pattern (FPN) noise in video data described in [1]
%
% [1] M. Maggioni, E. Sanchez-Monge, A. Foi, "Joint removal of random and
%     fixed-pattern noise through spatiotemporal video filtering", IEEE 
%     Transactions on Image Processing, vol.23, no.10, pp. 4282-4296, Oct. 2014
%     http://doi.org/10.1109/TIP.2014.2345261
% 
% [2] M. Maggioni, G. Boracchi, A. Foi, and K. Egiazarian, "Video denoising,
%     deblocking, and enhancement through separable 4-D nonlocal spatiotem-
%     poral transforms," IEEE Transactions on Image Processing, vol. 21,
%     no. 9, pp. 3952-3966, Sep. 2012.  http://doi.org/10.1109/TIP.2012.2199324
%
%
%  FUNCTION INTERFACE:
%
%  y_est = RF3D(z, sigma_rnd, sigma_fpn, psd_rnd, psd_fpn, transform_2D_name, erf3d, do_wiener, profile, print_to_screen)
%
%  INPUTS:
%     1) z               (3D array) : noisy video.
%     2) sigma_rnd         (single) : Scaling factor for the normalized
%                                     root PSDs of random noise component
%                                     (RND). Use sigma_rnd=-1 or
%                                     sigma_rnd=[] if unknown. 
%     3) sigma_fpn         (single) : Scaling factor for the normalized
%                                     root PSDs of fixed-pattern noise 
%                                     component (FPN). Use sigma_fpn=-1 or
%                                     sigma_fpn=[] if unknown. 
%     4) psd_rnd           (single) : normalized PSD of the random
%                                     noise component (RND). Use
%                                     psd_rnd=ones(8) for AWGN.
%     5) psd_fpn           (single) : normalized PSD of the fixed-pattern
%                                     noise component (FPN). Use
%                                     psd_rnd=ones(8) for AWGN.
%     6) transform_2D_name   (char) : name of the spatial transform used in
%                                     the filtering as well as for the PSDs
%                                     definition.
%     7) erf3d            (logical) : enable enhanced fixed-pattern noise
%                                     suppression (E-RF3D)
%                                     (optional, default is 1).
%     8) do_wiener        (logical) : perform collaborative wiener filtering
%                                     (optional, default is 1).
%     9) profile             (char) : 'lc' --> 'low complexity' profile
%                                   : 'np' --> 'normal' profile
%                                     'hc' --> 'high complexity' profile 
%                                     (optional, default is 'np').
%     10) print_to_screen (logical) : 0 --> do not print output information
%                                     1 --> print information to screen
%                                     (optional, default is 0).
%
%   Only z is required, all other inputs are optional. Optional inputs take
%   their default value when set to 'empty' [] .
%
%
%  OUTPUTS:
%     1) y_est           (3D array) : denoised video
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2011-2020    All rights reserved.
% This work should only be used for nonprofit purposes.
% Any unauthorized use of the provided software and files for industrial
% or profit-oriented activities is expressively prohibited.
% By using these files, you implicitly agree to all the terms of the
% TUT limited license included in the package and available at
% http://www.cs.tut.fi/~foi/legal_notice.html
% Please read the TUT limited license before you proceed with using these
% files.
%
% AUTHORS:
%     Enrique Sanchez-Monge  < esm _at_ noiselessimaging.com >
%     Matteo Maggioni 
%     Alessandro Foi  < alessandro.foi _at_ tuni.fi >
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Check arguments
%%%%

% If no transform operator is given, DCT is assumed by default
if ~exist('transform_2D_name','var') || isempty(transform_2D_name)
    transform_2D_name = 'dct';
    warning('Argument "transform_2D_name" is required but was omitted. Assuming DCT.');
end

% Enable Enhanced FPN suppression (E-RF3D) by default
if ~exist('erf3d','var') || isempty(erf3d)
    erf3d = true; 
end

% Enable Wiener Filtering by default
if ~exist('do_wiener','var') || isempty(do_wiener)
    do_wiener = 1;
end

% Quality/complexity profile selection
if ~exist('profile','var') || isempty(profile)
    profile = 'np'; % 'np' --> Normal Profile (balanced quality, default)
                    % 'lc' --> Low Complexity Profile (fast, lower quality)
end

% Check whether to dump information to the screen or remain silent
if ~exist('print_to_screen','var')||isempty(print_to_screen)
    print_to_screen = 0;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Algorithm parameters
%%%%

% Hard-thresholding (HT) parameters
N                   = 8;   % Block size NxN used for the HT (must be N=8)
Nstep               = 4;   % Pixels separating each processed reference block in HT
lambda_thr3D        = 2.7; % HT parameter for the 3D coefficient shrinkage (lambda in [1])

% Wiener filtering (WF) parameters
N_wiener            = N;   % Block size (must be N_wiener=8)
Nstep_wiener        = 4;   % Pixels separating each processed reference block in WF

% Number of frames spanned by the spatiotemporal volumes (both in HT and WF). 
h_plus              = 4;   % Number of future frames (must be <=10)
h_minus             = 4;   % Number of past frames (must be <=10)

% Select motion estimation strategy
searchType          = 3;   % searchType = 0; Full Search (as described in [2])
                           % searchType = 1; Fast Search (as described in [2])
                           % searchType = 2; Multi-Scale Full search
                           % searchType = 3; Multi-Scale Fast search

% Transform operators applied to the spatiotemporal volumes. The transforms
% can be 'dct', 'dst', 'hadamard', or anything that is listed by 'help wfilters').
% NOTE: the 2D spatial transform "transform_2D_name" should be the same as 
% that defining the PSDs psd_rnd and psd_fpn.
transform_2D_HT_name     = transform_2D_name; % 2D spatial transform used in HT
transform_2D_Wiener_name = transform_2D_name; % 2D spatial transform used in WF
transform_3rd_dim_name   = 'dct';             % 1D temporal transform applied on the 3rd dimension of the volumes both in HT and WF.

% Parameters overridden by the low complexity profile, i.e. the temporal 
% extent of the volumes is shotened and the steps are enlarged. 
if strcmpi(profile, 'lc') == 1
    h_plus              = 2;
    h_minus             = 2;
    Nstep               = 6;
    Nstep_wiener        = 5;
elseif strcmpi(profile, 'hc') == 1
    Nstep               = 3;
    Nstep_wiener        = 3;
    searchType          = 2;
end

% Fall back to white noise (flat PSD) in case the PSDs are not given
if ~exist('psd_rnd','var') || isempty(psd_rnd)
    psd_rnd = ones(N,'single');
    warning('Argument "psd_rnd" is required but was omitted. Assuming flat (white) PSD for the random noise component.');
end
if ~exist('psd_fpn','var') || isempty(psd_fpn)
    psd_fpn = ones(N,'single');
    warning('Argument "psd_fpn" is required but was omitted. Assuming flat (white) PSD for the fixed-pattern noise component.');
end

% In WF we use same PSDs used in HT (also the 2D transforms should be the same)
psd_rnd_wiener = psd_rnd;
psd_fpn_wiener = psd_fpn;

% Scaling factors for the normalized root PSDs of the RND and FPN
% If unknown, a value equal to -1 can be used to enable automatic noise
% estimation as in [1]. This is also the default behavior.
if ~exist('sigma_rnd','var') || isempty(sigma_rnd)
    sigma_rnd = -1;
    warning('Argument "sigma_rnd" was omitted. Enabling automatic noise estimation.');
end
if ~exist('sigma_fpn','var') || isempty(sigma_fpn)
    sigma_fpn = -1;
    warning('Argument "sigma_fpn" was omitted. Enabling automatic noise estimation.');

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Enhanced FPN suppression parameters
%%%%

% If sigma_fpn=0 then there is no FPN. In this case, E-RF3D cannot bring
% any improvement and it is therefore disabled.
if sigma_fpn==0 && erf3d
    erf3d = false;
    warning('No FPN present (sigma_fpn=0). Disabling E-RF3D.');
end

if erf3d
    motion_thrFPN = 0.1;  % Motion detection threshold used to estimate FP.
                          % Only frames having absolute mean motion bigger
                          % than the threshold are used in the FP estimation.
                          % A value equal to -1 disables the rule, and thus
                          % all frames are always included.
    lambda_thrFPN = 2;    % Threshold for outliers rejection during FP estimation.
                          % Values larger than the threshold are considered as
                          % part of the signal and thus rejected during the
                          % noise FP estimation.
else
    motion_thrFPN = 10e6; % A large value disables the enhanced FPN suppression
                          % because no frame is included in the FP estimation
    lambda_thrFPN = 0;    % A value equal to 0 disables the enhanced FPN 
                          % suppression because all values in the FP are
                          % considered as outliers and are thus rejected.
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Note: touch below this point only if you know what you are doing!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  Check variable sizes
%%%%

% Check video size
if ndims(z)>3 
    error('Invalid inuput video dimension. Only grayscale videos are allowed.');
end

% Check block size (currently only 8x8 size is supported)
if N~=8 || N_wiener~=8
    error('Block size not allowed. Blocks must have size 8x8.')
end

% Check PSDs size (currently only 8x8 size is supported)
if ~ismatrix(psd_rnd) || ~ismatrix(psd_fpn) || ...
        ~isequal(size(psd_rnd),[8 8]) || ~isequal(size(psd_fpn),[8 8])
    error('PSDs size not allowed. Both PSDs must be 2D array of size 8x8.')
end

% Check temporal extent of spatiotemporal volumes
if h_plus>10 || h_minus>10
    warning('Exceeded maximum temporal extent for the spatiotemporal volumes. The extent will be reduced to the maximum allowed value.');
    h_plus  = max(0,min(10,h_plus));
    h_minus = max(0,min(10,h_minus));
end
H = h_plus + h_minus + 1; %% extent of spatiotemporal volumes
assert(H<=21);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Create transform matrices
%%%%

% 2D spatial transforms
decLevel   = 0; %% dec. levels of the dyadic wavelet 2D transform for blocks (0 means full decomposition, higher values decrease the dec. number)
[Tfor, ~]  = getTransfMatrix(N, transform_2D_HT_name, decLevel);
[TforW, ~] = getTransfMatrix(N_wiener, transform_2D_Wiener_name, 0);
Tfor  = single(Tfor);
TforW = single(TforW);

% 1D temporal transform
if (strcmpi(transform_3rd_dim_name, 'dct') == 1) || (strcmpi(transform_3rd_dim_name, 'dst') == 1)
    forcePow2GroupLength = false;
else
    % restrict group size to be a power of 2 (applies in, e.g., Haar decomposition)
    forcePow2GroupLength = true;
end
Tfor3 = zeros(H,H,H,'single');
for h = 1:H
    if  forcePow2GroupLength && floor(log2(h))~=log2(h)
        Tfor3(1:h,1:h, h) = eye(h);
    else
        [Tfor3rd, ~]   = getTransfMatrix(h, transform_3rd_dim_name, 0);
        Tfor3(1:h,1:h, h) = single(Tfor3rd);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Filtering
%%%%

if print_to_screen
    fprintf('Filtering video of resolution %dx%dpx and %d frames \n', size(z,1), size(z,2), size(z,3));
    if sigma_rnd<0 || sigma_fpn<0
        fprintf('\tAdaptive noise estimation enabled \n');
    else
        fprintf('\tNoise scaling factors: sigma_rnd=%.1f and sigma_fpn=%.1f \n', sigma_rnd, sigma_fpn);
    end
    if do_wiener==1
        fprintf('\tWiener filtering enabled \n');
    else
        fprintf('\tWiener filtering disabled \n');
    end
    if erf3d==1
        fprintf('\tEnhanced fixed-pattern suppression enabled \n');
    else
        fprintf('\tEnhanced fixed-pattern suppression disabled \n');
    end
end

[ y_est ] = RF3D_8_mex( single(z), Nstep, Nstep_wiener, single(lambda_thr3D),...
    single(lambda_thrFPN), single(motion_thrFPN), h_minus, h_plus,...
    Tfor,TforW,Tfor3, logical(forcePow2GroupLength), single(sigma_rnd), single(sigma_fpn), single(psd_rnd),...
    single(psd_fpn), single(psd_rnd_wiener), single(psd_fpn_wiener), single(searchType), logical(do_wiener));

if ~isequal(class(z),class(y_est))
    y_est = cast(y_est,'like',z);
end

return;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some auxiliary functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [Tforward, Tinverse] = getTransfMatrix (N, transform_type, dec_levels)
%
% Create forward and inverse transform matrices, which allow for perfect
% reconstruction. The forward transform matrix is normalized so that the
% l2-norm of each basis element is 1.
%
% [Tforward, Tinverse] = getTransfMatrix (N, transform_type, dec_levels)
%
%  INPUTS:
%
%   N               --> Size of the transform (for wavelets, must be 2^K)
%
%   transform_type  --> 'dct', 'dst', 'hadamard', or anything that is
%                       listed by 'help wfilters' (bi-orthogonal wavelets)
%                       'DCrand' -- an orthonormal transform with a DC and all
%                       the other basis elements of random nature
%
%   dec_levels      --> If a wavelet transform is generated, this is the
%                       desired decomposition level. Must be in the
%                       range [0, log2(N)-1], where "0" implies
%                       full decomposition.
%
%  OUTPUTS:
%
%   Tforward        --> (N x N) Forward transform matrix
%
%   Tinverse        --> (N x N) Inverse transform matrix
%

if exist('dec_levels','var') ~= 1
    dec_levels = 0;
end

if N == 1
    Tforward = 1;
elseif strcmpi(transform_type, 'hadamard') == 1
    Tforward    = hadamard(N);
elseif (N == 8) && strcmpi(transform_type, 'bior1.5')==1 % hardcoded transform so that the wavelet toolbox is not needed to generate it
    Tforward =  [ 0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274;
        0.219417649252501   0.449283757993216   0.449283757993216   0.219417649252501  -0.219417649252501  -0.449283757993216  -0.449283757993216  -0.219417649252501;
        0.569359398342846   0.402347308162278  -0.402347308162278  -0.569359398342846  -0.083506045090284   0.083506045090284  -0.083506045090284   0.083506045090284;
        -0.083506045090284   0.083506045090284  -0.083506045090284   0.083506045090284   0.569359398342846   0.402347308162278  -0.402347308162278  -0.569359398342846;
        0.707106781186547  -0.707106781186547                   0                   0                   0                   0                   0                   0;
        0                   0   0.707106781186547  -0.707106781186547                   0                   0                   0                   0;
        0                   0                   0                   0   0.707106781186547  -0.707106781186547                   0                   0;
        0                   0                   0                   0                   0                   0   0.707106781186547  -0.707106781186547];
elseif (N == 8) && strcmpi(transform_type, 'dct')==1 % hardcoded transform so that the signal processing toolbox is not needed to generate it
    Tforward = [ 0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274   0.353553390593274;
        0.490392640201615   0.415734806151273   0.277785116509801   0.097545161008064  -0.097545161008064  -0.277785116509801  -0.415734806151273  -0.490392640201615;
        0.461939766255643   0.191341716182545  -0.191341716182545  -0.461939766255643  -0.461939766255643  -0.191341716182545   0.191341716182545   0.461939766255643;
        0.415734806151273  -0.097545161008064  -0.490392640201615  -0.277785116509801   0.277785116509801   0.490392640201615   0.097545161008064  -0.415734806151273;
        0.353553390593274  -0.353553390593274  -0.353553390593274   0.353553390593274   0.353553390593274  -0.353553390593274  -0.353553390593274   0.353553390593274;
        0.277785116509801  -0.490392640201615   0.097545161008064   0.415734806151273  -0.415734806151273  -0.097545161008064   0.490392640201615  -0.277785116509801;
        0.191341716182545  -0.461939766255643   0.461939766255643  -0.191341716182545  -0.191341716182545   0.461939766255643  -0.461939766255643   0.191341716182545;
        0.097545161008064  -0.277785116509801   0.415734806151273  -0.490392640201615   0.490392640201615  -0.415734806151273   0.277785116509801  -0.097545161008064];
elseif (N == 8) && strcmpi(transform_type, 'dst')==1 % hardcoded transform so that the PDE toolbox is not needed to generate it
    Tforward = [ 0.161229841765317   0.303012985114696   0.408248290463863   0.464242826880013   0.464242826880013   0.408248290463863   0.303012985114696   0.161229841765317;
        0.303012985114696   0.464242826880013   0.408248290463863   0.161229841765317  -0.161229841765317  -0.408248290463863  -0.464242826880013  -0.303012985114696;
        0.408248290463863   0.408248290463863                   0  -0.408248290463863  -0.408248290463863                   0   0.408248290463863   0.408248290463863;
        0.464242826880013   0.161229841765317  -0.408248290463863  -0.303012985114696   0.303012985114696   0.408248290463863  -0.161229841765317  -0.464242826880013;
        0.464242826880013  -0.161229841765317  -0.408248290463863   0.303012985114696   0.303012985114696  -0.408248290463863  -0.161229841765317   0.464242826880013;
        0.408248290463863  -0.408248290463863                   0   0.408248290463863  -0.408248290463863                   0   0.408248290463863  -0.408248290463863;
        0.303012985114696  -0.464242826880013   0.408248290463863  -0.161229841765317  -0.161229841765317   0.408248290463863  -0.464242826880013   0.303012985114696;
        0.161229841765317  -0.303012985114696   0.408248290463863  -0.464242826880013   0.464242826880013  -0.408248290463863   0.303012985114696  -0.161229841765317];
elseif strcmpi(transform_type, 'dct') == 1
    Tforward    = dct(eye(N));
elseif strcmpi(transform_type, 'dst') == 1
    Tforward    = dst(eye(N));
elseif strcmpi(transform_type, 'DCrand') == 1
    x = randn(N); x(1:end,1) = 1; [Q,~] = qr(x);
    if (Q(1) < 0)
        Q = -Q;
    end
    Tforward = Q';
else %% a wavelet decomposition supported by 'wavedec'
    %%% Set periodic boundary conditions, to preserve bi-orthogonality
    dwtmode('per','nodisp');
    
    Tforward = zeros(N,N);
    for i = 1:N
        Tforward(:,i)=wavedec(circshift([1 zeros(1,N-1)],[dec_levels i-1]), log2(N), transform_type);  %% construct transform matrix
    end
end

%%% Normalize the basis elements
Tforward = (Tforward' * diag(sqrt(1./sum(Tforward.^2,2))))';

%%% Compute the inverse transform matrix
Tinverse = inv(Tforward);

return;

