clear;clc;close all;
testdataset = 1;      
which_case = 'case1';
p_subspace = 10;
% %--------------------------
% testdataset = 2;
% which_case = 'case2';
% p_subspace = 10;
% %--------------------------
% testdataset = 1;
% which_case = 'case3';
% p_subspace = 10;
% %--------------------------
%  load clean image
%  Regarding the generation of the  clean image, see datasets\gen_data_sets.m
if testdataset==1  
     scene_name = 'paviaU';
     load('G:\PythonDemo\Test\HSI_SSFTTU\data\PaviaU.mat')
%          scene_name = 'Indian';
%      load('G:\PythonDemo\Test\HSI_SSFTTU\data\Indian_pines_corrected.mat')
%     scene_name = 'Salinas_corrected';
%      load('G:\PythonDemo\Test\HSI_SSFTTU\data\Salinas_corrected.mat')
else 
    scene_name = 'WashingtonDC';
    load img_clean_dc_withoutNormalization.mat;
    
end
fprintf(['\n','\n','Test dataset: ',scene_name,'  ',which_case,'\n']);

% img_clean = indian_pines_corrected;
% img_clean = salinas_corrected;
img_clean = paviaU;
[row, column, band] = size(img_clean);
N=row*column;

switch which_case
    case {'case1','case2'}       
        %Before adding noise, the gray values of each HSI band are normalized to [0,1].
        for i =1: band
            y = img_clean(:, :, i) ;
            max_y = max(y(:));
            min_y = min(y(:));
            y =  (y - min_y)./ (max_y - min_y);
            img_clean(:, :, i) = y;
        end
    otherwise %without normalization
        %do nothing     
end

%Add noise
switch which_case
    case 'case1'
        %--------------------- Case 1 --------------------------------------
        
        % zero-mean gaussian noise is added to all the bands of the Washington DC Mall
        % and Pavia city center data
        
        noise_type='additive';
        iid = 1;
        i_img=1;
        sigma = 0;randn('seed',0);
        
        %generate noisy image
        noise = sigma.*randn(size(img_clean));
        img_noisy=img_clean+noise;
        
    case 'case2'
        %---------------------  Case 2 ---------------------
        
        % Different variance zero-mean Gaussian noise is added to
        % each band of the two HSI datasets
        % The std values are randomly selected from 0 to 0.1.
         noise_type='additive';
         iid = 0;
        rand('seed',0);
        sigma = rand(1,band)*0.1;
        randn('seed',0);
        noise= randn(size(img_clean));
        for cb=1:band
            noise(:,:,cb) = sigma(cb)*noise(:,:,cb);
            
        end
        img_noisy=img_clean+noise;
            
    case 'case3'
        %  ---------------------  Case 3: Poisson Noise ---------------------
         noise_type='poisson';
          iid = NaN; % noise_type is set to 'poisson'
        img_wN = img_clean;
        
        snr_db = 15;
        snr_set = exp(snr_db*log(10)/10);
        
        for i=1:band
            img_wNtmp=reshape(img_wN(:,:,i),[1,N]);
            img_wNtmp = max(img_wNtmp,0);
            factor = snr_set/( sum(img_wNtmp.^2)/sum(img_wNtmp) );
            img_wN_scale(i,1:N) = factor*img_wNtmp;
            % Generates random samples from a Poisson distribution
            img_wN_noisy(i,1:N) = poissrnd(factor*img_wNtmp);
        end
        
        img_noisy = reshape(img_wN_noisy', [row, column band]);
        img_clean = reshape(img_wN_scale', [row, column band]);        
end
M=ones(size(img_noisy));
img_noisy_nan=img_noisy;
bands_strp=60:63;
for ib =  bands_strp    
    if ib == 60      
        loc_strp = ceil(rand(1,20)*column);
        switch which_case
            case 'case1'
                loc_strp = [loc_strp, 80:145];%loc_strp = [loc_strp, 200:210];
                loc_strp = [loc_strp, 120:140]; %simulate a hole
            case 'case2'
                loc_strp = [loc_strp, 20:40];
                loc_strp = [loc_strp, 160:175]; %simulate a hole
            case 'case3'
                loc_strp = [loc_strp, 70:90];
                loc_strp = [loc_strp, 150:160]; %simulate a hole
        end
    end
    %         img_noisy_nan(:,loc_strp,ib)=zeros(row,size(loc_strp,2))*NaN;
    img_noisy(:,loc_strp,ib)=zeros(row,size(loc_strp,2));
    M(:,loc_strp,ib)=zeros(row,size(loc_strp,2));
end

addpath('..\Hyreconstruction\FastHyDe');
[img_fasthyde, time_fasthyde] = FastHyIn(img_noisy, M, noise_type, iid, p_subspace);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%  show original and reconstructed data   %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Y_ori = reshape(img_noisy, [],band)';
% Y_clean  = reshape(img_clean, [],band)';
% Y_fasthyde  = reshape(img_fasthyde, [],band)';
%  
% figure(100);
% set(gcf,'outerposition',get(0,'screensize'))
% i=1;
% time_fig=[];
% title_fig={};
% figdata=[];
% %figdata(1:band,1:N,i)=reshape(img_clean, N, band)';         i=i+1;    time_fig=[time_fig,0];                 title_fig={title_fig{:},'Clean '};
% %figdata(:,:,i)= reshape(img_noisy, N, band)';               i=i+1;    time_fig=[time_fig,0];                 title_fig={title_fig{:},'Noisy band'};
% figdata(:,:,i)= reshape(img_fasthyde, N, band)';       
% i=i+1;    
% time_fig=[time_fig,round(time_fasthyde)];         
% title_fig={title_fig{:},'FastHyIn'};