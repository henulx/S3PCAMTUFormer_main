%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  DemoRF3D (ver. 1.1, 25 February 2020)
%
%  DemoRF3D demonstrates the algorithm for joint attenuation of random
%  (RND) and Fixed-Pattern (FPN) noise described in the paper:
%
% [1] M. Maggioni, E. Sanchez-Monge, A. Foi, "Joint removal of random and
%     fixed-pattern noise through spatiotemporal video filtering", IEEE
%     Transactions on Image Processing, vol.23, no.10, pp. 4282-4296, Oct. 2014
%     http://doi.org/10.1109/TIP.2014.2345261
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2011-2020    All rights reserved.
% This work should only be used for nonprofit purposes.
% Any unauthorized use of the provided software and files for industrial
% or profit-oriented activities is expressively prohibited.
% By using these files, you implicitly agree to all the terms of the
% TUT limited license, as specified in the document
% LEGAL_NOTICE.txt included in this package, and online at
% http://www.cs.tut.fi/~foi/GCF-BM3D/legal_notice.html
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
%%%%  MODIFIABLE DEMO PARAMETERS
%%%%

% Choose experiment type
experiment_type = 0;       % Available options:
%  0 --> denoising of real noisy video acquired by a
%        FLIR Tau 320 microbolometer camera
%  1 --> denoising of video corrupted by synthetic noise
%  2 --> denoising of user provided video sequence

% Path to video file (sequences available at http://www.cs.tut.fi/~foi/GCF-BM3D)
file_name = '';            % If empty, a prompt dialog will appear. The file
% type allowed to be selected depends on experiment_type.


% Noise characterization:
if experiment_type==0
    % Scaling factors for the normalized root PSD of the RND and FPN noise
    % components. By using value equal to -1  we enable automatic noise
    % estimation as in [1].
    sigma_rnd = -1;            % Scaling factor of random noise (RND)    
    sigma_fpn = -1;            % Scaling factor of fixed-pattern noise (FPN)
elseif experiment_type==1
    % Noise parameters used to generate the synthetic random and fixed-pattern
    % noise added to the noise-free videos used when experiment_type = 1.
    % These parameters apply IF AND ONLY IF experiment_type = 1.
    % Real noisy videos do not require generation of additional synthetic noise.
    sigma_rnd = 15;            % Scaling factor of random noise (must be >=0).
    sigma_fpn = 15;            % Scaling factor of fixed-pattern noise (FPN) (must be >=0).
    FPN_drift = 0.0;           % FPN drift (0<=FPN_drift<=1).
    % If equal to 0, the FPN is static (i.e. no drift).
    % If 0 < FPN_drift < 1, the FPN drifts at each frame.
    % If FPN_drift = 1, the FPN component changes completely
    % at every frame, thus degenerating into another RND component.
    % FPN_drift =  0  was used for Table I in [1].
    % FPN_drift =  0.01  was used for Table II in [1].
elseif experiment_type==2
    % Noise parameters for user defined video.  MODIFY THIS TO ADAPT TO
    % YOUR DATA (PSDs and scaling factors)
    % Scaling factors for the normalized root PSD of the RND and FPN noise
    % components. By using value equal to -1  we enable automatic noise
    % estimation as in [1].
    sigma_rnd = 15;               % Scaling factor of random noise. 
                                  % Use -1 if unknown.
                                  % Use 0 if no random noise component
                                  % present in video.
    sigma_fpn = 15;               % Scaling factor of fixed-pattern noise (FPN).
                                  % Use -1 if unknown.
                                  % Use 0 if no fixed-pattern noise component
                                  % present in video.
    % Normalized Power Spectral Densities (PSD) of the noise components.
    % These must be normalized with respect their highest frequency
    % coefficient. 
    psd_rnd   = ones(8,'single'); % User defined PSD for the RND component.
                                  % Use ones(8,'single') for AWGN
    psd_fpn   = ones(8,'single'); % User defined PSD for the FPN component. 
                                  % Use ones(8,'single') for AWGN
end


% RF3D parameters
profile           = 'np';  % Complexity profile:
%  'np' --> Normal Profile (balanced quality, default)
%  'lc' --> Low Complexity Profile (fast, lower quality)
%  'hc' --> High Complexity Profile (slow, higher quality)
transform_2D_name = 'dct'; % 2D block transform used in RF3D (it also defines the PSDs)
% RF3D accepts 'dct', 'dst', 'hadamard', as well
% as everything listed by 'help wfilters'.
% However, the same transform operator defines
% the PSDs of the noise. We only provide the
% PSDs for 'dct', 'bior1.5', or 'haar'.
do_wiener         = false;  % Wiener Filtering (true or false).
erf3d             = true;  % Enhanced FPN suppression (true or false).
print_to_screen   = true;  % Verbose mode (true or false).




% In all cases, the PSDs should be defined to the same 2D transform used
% in RF3D (i.e., transform_2D_name). In this package, we provide the PSDs
% for 'dct', 'bior1.5', and 'haar' transform for the FLIR Tau 320 camera
% nose model.
if experiment_type==0
    % Loading real data acquired by FLIR Tau 320 camera and the corresponding PSDs
    [z, psd_rnd, psd_fpn] = LoadRealDataFLIR(file_name, transform_2D_name);
     
elseif experiment_type==1
    % Loading noise-free and noisy synthetic data and the corresponding PSDs
    [y, z, psd_rnd, psd_fpn] = LoadSyntheticData(file_name, transform_2D_name, sigma_rnd, sigma_fpn, FPN_drift);
    
elseif experiment_type==2
    % MODIFY TO YOUR OWN NEEDS
    % example A) Load your own noiy sequence.
    [ z ] = LoadRealData( file_name );
    % example B) Add AWGN noise (RND and FPN) to your own noise-free sequence
    %   [ y ] = LoadRealData( file_name );
    %   z = y + randn(size(y))*sigma_rnd + repmat(randn(size(y,1),size(y,2)),[1 1 size(y,3)])*sigma_fpn;
else
    % Invalid experiment_type option
    error('Invalid "experiment_type" option. Plese, select 0, 1 or 2.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  RF3D FILTERING (NOTHING SHOULD BE CHANGED BELOW THIS POINT)
%%%%
% RF3D filtering execution (refer to RF3D.m for additional information).
% The PSDs of the noise are assumed to be known, whereas the scaling
% factors are always automatically estimated from the noisy data.
timeFirst = now;
y_est = RF3D(z, sigma_rnd, sigma_fpn, psd_rnd, psd_fpn, transform_2D_name, erf3d, do_wiener, profile, print_to_screen);
estimate_elapsed_time = 86400*(now-timeFirst);

% Showing objective (PSNR) and subjective results
fprintf('Filtering completed: %.1f sec (%.1f fps)', ...
    estimate_elapsed_time, size(z,3)/estimate_elapsed_time);
if experiment_type==1 && exist('y','var') && ~isempty(y)
    PSNR = 10*log10(255^2/mean((y(:)-y_est(:)).^2)); % the assumed signal peak is 255
    fprintf(', PSNR %.2f dB. \n', PSNR);
else
    fprintf('.\n');
end

implay(([z y_est]-min(y_est(:)))/(max(y_est(:))-min(y_est(:))))

