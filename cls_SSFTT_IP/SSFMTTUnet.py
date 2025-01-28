import PIL #PIL(Python Imaging Library)是一个用于打开、编辑和保存多种图像文件格式的库
import time
import torch
import torchvision#torch类型的数据
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init



def _weights_init(m):
    classname = m.__class__.__name__#获取变量m的类名
    if isinstance(m, nn.Linear) or isinstance(m,nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):#类继承nn.Module
    def __init__(self,fn):
        super().__init__()#初始化操作
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self,dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,dim,heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim*3, bias=True)# Wq,Wk,Wv for each vector, thats why *3
        self.nn1 = nn.Linear(dim,dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3,dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x,mask = mask)
            x = mlp(x)
        return x

NUM_CLASS = 9
# NUM_CLASS = 16

class Spectral(nn.Module):
    def __init__(self,L1):
        super(Spectral, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv2d(L1, 8, kernel_size=1, stride=1, padding=0),
        )

        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
        )

        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),
        )

        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(24, 8, kernel_size=1, stride=1, padding=0),
        )

        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv15 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x11 = self.conv11(x)

        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x11)

        x13 = torch.cat((x11, x12), dim=1)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)

        x15 = self.batch_norm14(x15)
        x16 = self.conv15(x15)
        return x16


class Spatial(nn.Module):
    def __init__(self, L2):
        super(Spatial, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv2d(L2, 8, kernel_size=3, stride=1, padding=1),
        )

        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv12 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
        )

        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv13 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
        )

        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.conv14 = nn.Sequential(
            nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1),
        )

        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv15 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x11 = self.conv11(x)

        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x11)

        x13 = torch.cat((x11, x12), dim=1)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x15 = self.batch_norm14(x15)
        x16 = self.conv15(x15)
        return x16

class SSFTTUnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSFTTUnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.spe = Spectral(L1=30)
        self.spa = Spatial(L2=30)

        #Spectral Attention Block
        self.query_conv = nn.Conv2d(in_channels=30, out_channels=30 // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=30, out_channels=30 // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        #Multi-Spatial Attention Block
        self.twoconv2d_featuresSize3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(3,3)),#修改in_channels=8*28为30
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.twoconv2d_featuresSize5 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(5, 5)),  # 修改in_channels=8*28为30
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.twoconv2d_featuresSize7 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=64, kernel_size=(7, 7)),  # 修改in_channels=8*28为30
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 7), stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.convm0 = nn.Sequential(
            nn.Conv2d(64, 9, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )##UP
        # self.convm0 = nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=1, padding=0),
        #     nn.Sigmoid(),
        # )##IP
        # unmixing module
        # self.unmix_encoder = nn.Sequential(
        #     nn.Conv2d(30, 15, kernel_size=(1,1), stride=1, padding=0),
        #     # 步幅：卷积核经过输入特征图的采样间隔，希望减小输入参数的数目，减少计算量
        #     # 填充：填充：在输入特征图的每一边添加一定数目的行列，使得输出的特征图的长、宽 = 输入的特征图的长、宽
        #     nn.BatchNorm2d(15),
        #     # 在卷积层之后和激活函数之前，BatchNorm2d的主要作用是通过减少内部协变量偏移来加速网络的训练，并提高模型的泛化能力。
        #     nn.ReLU(),
        #     nn.Conv2d(15, 7, kernel_size=(1,1), stride=1, padding=0),
        #     nn.BatchNorm2d(7),
        #     nn.ReLU(),
        #     nn.Conv2d(7, num_classes, kernel_size=(1,1), stride=1, padding=0)
        # )  # 编码器，包含3个1*1的卷积
        self.unmix_encoder = nn.Sequential(
            nn.Conv2d(30, 15, kernel_size=(3, 3), stride=1, padding=1),
            # 步幅：卷积核经过输入特征图的采样间隔，希望减小输入参数的数目，减少计算量
            # 填充：填充：在输入特征图的每一边添加一定数目的行列，使得输出的特征图的长、宽 = 输入的特征图的长、宽
            nn.BatchNorm2d(15,affine=True),
            # 在卷积层之后和激活函数之前，BatchNorm2d的主要作用是通过减少内部协变量偏移来加速网络的训练，并提高模型的泛化能力。
            nn.ReLU(),

            nn.Conv2d(15, 7, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(7,affine=True),
            nn.ReLU(),

            nn.Conv2d(7, num_classes, kernel_size=(3, 3), stride=1, padding=1),
            nn.Softmax(dim=1)
        )  # 编码器，包含3个1*1的卷积
        self.unmix_decoder = nn.Sequential(
            nn.Conv2d(num_classes, 60, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.unmix_decoder_nonlinear = nn.Sequential(
            nn.Conv2d(30 * 2, 30, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(30, 30, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )  # 3*3大小卷积核，

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),#self.L==4
                                      requires_grad=True) # Tokenization parameters
        # self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),  # self.L==4
        #                              requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                      requires_grad=True) # Tokenization parameters
        # self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
        #                              requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        #self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens+1), dim))
        #self.pos_embedding = nn.Parameter(torch.empty(1, 13, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, 13, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        # self.nn1 = nn.Linear(640, num_classes)  # 640*16,IP或者SA
        self.nn1 = nn.Linear(505, num_classes)  #388*9,UP

        torch.nn.init.xavier_normal_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):

        # Spectral Attention Block
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = energy.softmax(dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        RX = self.gamma * out + x#获得的光谱注意力块RX

        #Image unmixing Method1
        x1 = self.spe(x)
        x2 = self.spa(x)
        x3 = torch.cat((x1, x2), dim=1)
        abu = self.convm0(x3)
        #Image unmixing Method2
        #abu = self.unmix_encoder(x)  # 编码器()求出丰度矩阵
        re_unmix = self.unmix_decoder(abu) #解码部分第一次卷积（1*1）
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)  #解码部分的两次卷积（1*1和1*1）
        #ReconstructedImage = torch.cat([re_unmix,re_unmix_nonlinear],1)
        abu = abu.abs()  # 非负，解混的丰度限定
        abu = abu / abu.sum(1).unsqueeze(1)  # 和为1，解混的丰度限定

        # reshape abu
        feature_abu = self.conv(abu)  # 对求得的丰度矩阵进行3*3的卷积 torch.Size([64, 16, 6, 6])
        abu_v = feature_abu.reshape(x.shape[0], -1)#size:[64, 576]

        #multiscale two-dimensional convolution
        cx = RX[:,:,:,:]#3分支2D卷积神经网络卷积核大小为5，torch.Size([64, 30, 13, 13])
        cx = self.twoconv2d_featuresSize5(cx)
        bx = RX[:,:,:,:]#3分支2D卷积神经网络卷积核大小为4
        bx = self.twoconv2d_featuresSize7(bx)
        RX = self.twoconv2d_featuresSize3(RX)##3分支2D卷积神经网络卷积核大小为3

        RX = rearrange(RX,'b c h w -> b (h w) c')     #[64,64,9,9]->[64,81,64]
        cx = rearrange(cx, 'b c h w -> b (h w) c')  #[64,64,5,5]->[64,25,64]
        bx = rearrange(bx, 'b c h w -> b (h w) c')  # [64,64,7,7]->[64,16,64]

        wa = rearrange(self.token_wA, 'b h w -> b w h') # Transpose[64,4,64]->[64,64,4]
        A = torch.einsum('bij,bjk->bik', RX, wa)#x*Wa  [64,81,64]*[64,64,4]=[64,81,4]（又有问题了）
        A = rearrange(A, 'b h w -> b w h') #Transpose:[64,4,81]
        A = A.softmax(dim=-1)#softmax:[64,4,81]

        B = torch.einsum('bij,bjk->bik', cx, wa)  # cx*Wa  [64,25,64]*[64,64,4]=[64,25,4]
        B = rearrange(B, 'b h w -> b w h')  # Transpose:[64,4,25]
        B = B.softmax(dim=-1)  # softmax:[64,4,25]

        C = torch.einsum('bij,bjk->bik', bx, wa)  # cx*Wa  [64,16,64]*[64,64,4]=[64,16,4]
        C = rearrange(C, 'b h w -> b w h')  # Transpose:[64,4,16]
        C = C.softmax(dim=-1)  # softmax:[64,4,16]


        VV = torch.einsum('bij,bjk->bik', RX, self.token_wV)#softmax(x*Wa)T*X:[64,81,64]*[64,64,64]=[64,81,64]
        T = torch.einsum('bij,bjk->bik', A, VV)#[64,4,81]*[64,81,64]=[64,4,64]

        VV1 = torch.einsum('bij,bjk->bik', cx, self.token_wV)  # softmax(x*Wa)T*X:[64,25,64]*[64,64,64]=[64,25,64]
        T1 = torch.einsum('bij,bjk->bik', B, VV1)  # [64,4,25]*[64,25,64]=[64,4,64]

        VV2 = torch.einsum('bij,bjk->bik', bx, self.token_wV)  # softmax(x*Wa)T*X:[64,16,64]*[64,64,64]=[64,16,64]
        T2 = torch.einsum('bij,bjk->bik', C, VV2)  # [64,4,16]*[64,16,64]=[64,4,64]

        cls_tokens = self.cls_token.expand(x.shape[0], -1,-1)#size(64,1,64)
        TransformerInput = torch.cat((cls_tokens, T, T1, T2), dim=1)#size(64,13,64)
        TransformerInput += self.pos_embedding#size(64,13,64)
        TransformerInput = self.dropout(TransformerInput)#size(64,13,64)
        TransformerInput = self.transformer(TransformerInput,mask)#size(64,13,64)
        TransformerInput = self.to_cls_token(TransformerInput[:,0])#size(64,64)
        feature_fuse = torch.cat([abu_v, TransformerInput], dim=1)#abu_v:size(64*576)设置stride=2, x:size(64*64)=>feature_fuse:size(64*640)
        LastFeature = self.nn1(feature_fuse)#size(64*640) * size(640*16) =>size(64*16)

        #return LastFeature
        return LastFeature, re_unmix, re_unmix_nonlinear #size x1([64,16])
        #return re_unmix_nonlinear  #size([64, 30, 13, 13])
        #return re_unmix    #size([64, 60, 13, 13])
        #return abu_v #当stride=2时，abu_v的大小为1*576，当stride = 6时，abu_v的大小为1*64


if __name__ == '__main__':
        model = SSFTTUnet()#基于高光谱图像解混的空谱联合符号化Transformer的分类网络
        model.eval()
        print(model)
        #input = torch.randn(100, 200, 13, 13)
        input = torch.randn(32,30,15,15)
        y = model(input)
        #print(y.size())