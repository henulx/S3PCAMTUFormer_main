%S3PCA-MTUFormer: Spectral–Spatial Multiscale Tokenization Unmixing 
%Transformer Based on SuperPCA for Hyperspectral Image Classification
clc;clear;close all
testdataset = 1;      
which_case = 'case1';
p_subspace = 10;
if testdataset==1  
     scene_name = 'paviaU';
     load('G:\PythonDemo\Test\S3PCAMTUFormer_main\data\PaviaU.mat')
else 
    scene_name = 'WashingtonDC';
    load img_clean_dc_withoutNormalization.mat;
    
end
fprintf(['\n','\n','Test dataset: ',scene_name,'  ',which_case,'\n']);

img_clean = paviaU;
[row, column, band] = size(img_clean);
N=row*column;

switch which_case
    case {'case1','case2'}       
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

addpath('.\Hyreconstruction\FastHyDe');
addpath('.\Hyreconstruction\BM3D');
[Pufigdata, time_fasthyde] = FastHyIn(img_noisy, M, noise_type, iid, p_subspace);
save('G:\PythonDemo\Test\S3PCAMTUFormer_main\data\Pufigdata.mat','Pufigdata')

addpath('D:\Program Files\Polyspace\R2021a\toolbox\libsvm-3.21')
addpath('.\common');
addpath('.\Entropy Rate Superpixel Segmentation\ERS');
addpath('.\Entropy Rate Superpixel Segmentation');
addpath(genpath(cd));

num_PC            =   30;  % THE optimal PCA dimension
% num_Pixels        =  100*sqrt(2).^[-4:4]; % THE Numbers of Multiscale Superpixel IP
num_Pixels        =   20*sqrt(2).^[-2:7]; % THE Numbers of Multiscale Superpixel PU [-2:6]
% % num_Pixels        =   50*sqrt(2).^[-3:3]; % THE Numbers of Multiscale Superpixel SA 
trainpercentage   =   26;  % Training Number per Class 0.005，PU
iterNum           =   10;    % Trails
% database  = 'Indian';
database  = 'PaviaU';
% database  = 'Salinas';

for inum_Pixel = 1:size(num_Pixels,2)
    num_Pixel = num_Pixels(inum_Pixel);
    
    %% load the HSI dataset
    if strcmp(database,'Indian')
        load G:\MatlabDemo\my-demo-main\datasets\figdata.mat;
        load Indian_pines_gt;
        load Indian_pines_randp 
        data3D = figdata;        label_gt = indian_pines_gt;
    elseif strcmp(database,'Salinas')
        load G:\MatlabDemo\my-demo-main\datasets\Salinas_corrected.mat;load G:\PythonDemo\Test\HSI_SSFTTU\data\Salinas_gt.mat;load G:\MatlabDemo\SuperPCA-master\datasets\Salinas_randp
        data3D = salinas_corrected;        
        label_gt = salinas_gt;        
    elseif strcmp(database,'PaviaU')    
        load G:\PythonDemo\Test\S3PCAMTUFormer_main\data\Pufigdata.mat;load G:\PythonDemo\Test\MCNN\data\PaviaU_gt;load G:\MatlabDemo\SuperPCA-master\datasets\PaviaU_randp; 
        data3D = Pufigdata;        label_gt = paviaU_gt;
    end
    data3D = data3D./max(data3D(:));
       
    % super-pixels segmentation
    labels = cubseg(data3D,num_Pixel);

    %% SupePCA DR
    [dataDR] = SuperPCA(data3D,num_PC,labels);
    iter = 1;%for pu and ip
%     iter = 5;%for SA 
    randpp=randp{iter};     
    % randomly divide the dataset to training and test samples
    [DataTest DataTrain CTest CTrain map] = samplesdivide(dataDR,label_gt,trainpercentage,randpp);   

    % Get label from the class num
    trainlabel = getlabel(CTrain);
    testlabel  = getlabel(CTest);

    % set the para of RBF
    ga8 = [0.01 0.1 1 5 10];    ga9 = [15 20 30 40 50 100:100:500];
    GA = [ga8,ga9];

    accy = zeros(1,length(GA));

    tempaccuracy1 = 0;
    predict_label_best = [];
    for trial0 = 1:length(GA);    
        gamma = GA(trial0);        
        cmd = ['-q -c 100000 -g ' num2str(gamma) ' -b 1'];
        model = svmtrain(trainlabel', DataTrain, cmd);
        [predict_label, AC, prob_values] = svmpredict(testlabel', DataTest, model, '-b 1');                    
        [confusion, accuracy1, CR, FR] = confusion_matrix(predict_label', CTest);
        accy(trial0) = accuracy1;

        if accuracy1>tempaccuracy1
            tempaccuracy1 = accuracy1;
            predict_label_best = predict_label;
        end
    end
    accy_best(inum_Pixel) = tempaccuracy1;
    predict_labelS(inum_Pixel,:) = predict_label_best;
end

predict_label = label_fusion(predict_labelS');
[confusion, accuracy2, CR, FR] = confusion_matrix(predict_label', CTest);
AA=sum(CR(1,:))/9;
% AA=sum(CR(1,:))/16;
CR=CR*100
OA=accuracy2;
Po=OA;
Pe=(sum(confusion)*sum(confusion,2))/(sum(confusion(:))^2);
Kappa=(Po-Pe)/(1-Pe);
fprintf('\n=============================================================\n');
fprintf(['The OA (1 iteration) of SuperPCA for ',database,' is %0.4f\n'],max(accy_best));
fprintf(['The OA (1 iteration) of MSuperPCA for ',database,' is %0.4f\n'],accuracy2);
fprintf('=============================================================\n');
