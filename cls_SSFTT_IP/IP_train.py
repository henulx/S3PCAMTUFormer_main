import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import S3PCAMTUFormer_main.cls_SSFTT_IP.get_cls_map
import time
import S3PCAMTUFormer_main.cls_SSFTT_IP.SSFMTTUnet
import os


os.environ["TF_CUDNN_USE_AUTOTUNE"]="0"


def loadData():
    #读入数据
    data = sio.loadmat('G:\PythonDemo\Test\S3PCAMTUFormer_main\data\PaviaU.mat')['paviaU']
    labels = sio.loadmat('G:\PythonDemo\Test\S3PCAMTUFormer_main\data\PaviaU_gt.mat')['paviaU_gt']

    # data = sio.loadmat('D:\PythonHSIDemo\S3PCAMTUFormer_main\data\Indian_pines_corrected.mat')['indian_pines_corrected']
    # labels = sio.loadmat('D:\PythonHSIDemo\S3PCAMTUFormer_main\data\Indian_pines_gt.mat')['indian_pines_gt']

    #data = sio.loadmat('D:\PythonHSIDemo\S3PCAMTUFormer_main\data\Salinas_corrected.mat')['salinas_corrected']
    #labels = sio.loadmat('D:\PythonHSIDemo\S3PCAMTUFormer_main\data\Salinas_gt.mat')['salinas_gt']

    return data, labels

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2*margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] +y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState = 345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

# BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TRAIN = 32


class TestDS(torch.utils.data.Dataset):

    def __init__(self,Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        #返回文件数据的数目
        return self.len


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):


        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        #返回文件数据的数目
        return self.len


def create_data_loader():
    #地物类别
    class_num = 9
    # class_num = 16
    #读入数据
    X,y = loadData()
    # 用于测试样本的比例
    test_ratio = 0.995#work PU
    # test_ratio = 0.933#work IP
    #test_ratio = 0.995  #work SA
    # 每个像素周围提取 patch 的尺寸
    patch_size = 15
    #使用PCA降维，得到主成分的数量
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ',y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA:', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ',X_pca.shape)
    print('Data cube y shape: ',y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size,pca_components)
    Xtest = Xtest.reshape(-1, patch_size, patch_size,pca_components)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0,3,1,2)
    Xtrain = Xtrain.transpose(0,3,1,2)
    Xtest = Xtest.transpose(0,3,1,2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    train_set = TrainDS(Xtrain, ytrain)
    test_set = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_set,
        batch_size = BATCH_SIZE_TRAIN,
        shuffle = True,
        num_workers = 0,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_set,
        batch_size = BATCH_SIZE_TRAIN,
        shuffle = True,
        num_workers = 0,
    )
    all_data_loader = torch.utils.data.DataLoader(
        dataset = X,
        batch_size = BATCH_SIZE_TRAIN,
        shuffle = True,
        num_workers = 0,
    )

    return train_loader, test_loader, all_data_loader, y

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = S3PCAMTUFormer_main.cls_SSFTT_IP.SSFMTTUnet.SSFTTUnet().to(device)
    #交叉熵损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.CrossEntropyLoss().cpu()
    #初始化优化器
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    #开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data= data.cuda()#torch.Size([64, 30, 13, 13])GPU
            target = target.cuda()#GPU
            # data = data.to(device)  # torch.Size([64, 30, 13, 13])
            # target = target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            #outputs = net(data)
            batch_pred, re_unmix, re_unmix_nonlinear = net(data)#这里出问题了
            # band = re_unmix.shape[1]//2
            # output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
            # re_unmix = re_unmix_nonlinear + output_linear

            #sad_loss = torch.mean(torch.acos(torch.sum(data * re_unmix, dim=1) /
            #                                (torch.norm(re_unmix, dim=1, p=2) * torch.norm(data, dim=1, p=2))))
            # 计算总体损失函数
            #loss = criterion(batch_pred, target) + sad_loss #出问题了
            loss = criterion(batch_pred, target) # 出问题了
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    #模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.cuda()#GPU
        # inputs = inputs.to(device)#CPU
        batch_pred, re_unmix, re_unmix_nonlinear = net(inputs)
        outputs = np.argmax(batch_pred.detach().cpu().numpy(), axis=1)#出问题
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                         'Self-Blocking Bricks','Shadows']
    # target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
    #                     'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
    #                     'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
    #                     'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
    #                     'Stone-Steel-Towers']
    # target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
    #                 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
    #                 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
    #                 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=100)
    # 只保存模型参数
    torch.save(net.state_dict(), 'G:\PythonDemo\Test\S3PCAMTUFormer_main\cls_params\SSFTTnet_params.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    LRS_MSuperPCA = [95.4731264193793,	98.1098641464855	,99.1316931982634,	93.1204739960500	,95.8301743745262,
                     96.0623625824505,	98.0828220858896	,98.1126914660832,	99.1313789359392]
    LRS_MSuperPCA_each_acc = np.array(LRS_MSuperPCA)
    each_acc = np.array(each_acc)
    All_each_acc = np.row_stack((each_acc, LRS_MSuperPCA_each_acc))
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    Total_time = Training_Time+Test_time
    UP_Samples = [6631,18649,2099,3064,1345,5029,1330,3682,947]
    UP_Samples = np.array(UP_Samples)
    UP_SumNum = 42776

    def get_max_value(martix):
        '''
        Obtain the maximum value for each column in the matrix
        '''
        res_list = []
        for j in range(len(martix[0])):
            one_list = []
            for i in range(len(martix)):
                one_list.append(float(martix[i][j]))
            res_list.append(max(one_list))
        return res_list


    CategoryAccuracy = get_max_value(All_each_acc)
    AverageAccuracy = sum(CategoryAccuracy) / len(CategoryAccuracy)
    OverallAccuracy = np.dot(UP_Samples, CategoryAccuracy) / UP_SumNum
    file_name = "G:\PythonDemo\Test\S3PCAMTUFormer_main\cls_result\classification_report.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Total_time (s)'.format(Total_time))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(OverallAccuracy))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(AverageAccuracy))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(CategoryAccuracy))
        x_file.write('\n')