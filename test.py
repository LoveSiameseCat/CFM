import timm
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import math
import copy

class EfficientNet(nn.Module):
    def __init__(self,pretrained=True):
        super(EfficientNet, self).__init__()
        '''
        scale_down: conv_stem, block1, block2, block3, block5
        '''
        m = timm.create_model('efficientnet_b4',pretrained=pretrained)
        m_ = OrderedDict()
        block_list = []
        for name,module in m.named_children():
            if name == 'blocks':
                for block in module.children():
                    block_list.append(block)
            m_[name]=module
        self.conv_stem = m_['conv_stem']
        self.bn1 = m_['bn1']
        self.act1 = m_['act1']

        self.block0 = block_list[0]
        self.block1 = block_list[1]
        self.block2 = block_list[2]
        self.block3 = block_list[3]
        self.block4 = block_list[4]
        self.block5 = block_list[5]
        self.block6 = block_list[6]
        self.conv_head = m_['conv_head']
        self.bn2 = m_['bn2']
        self.act2 = m_['act2']
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1792,2)
        self.drop = nn.Dropout(0.5)

        del m,m_,block_list

    def extract_fea(self,x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        feature = x
        return feature

    def forward(self,x):
        b,c,h,w = x.size()
        feature = self.extract_fea(x)
        x = self.pool(feature).view(feature.size(0),-1)
        x = self.drop(x)
        logits = self.fc(x)
        return feature,logits

class SCL(nn.Module):
    def __init__(self, dim=128, m=0.999):
        """
        dim: feature dimension (default: 128)
        m: momentum of updating key encoder (default: 0.999)
        """
        super(SCL, self).__init__()
        self.m = m
        self.dim = dim
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.alpha = 0.99
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # unused_parameter
        self.T = 0.2
        self.margin = 0.5
        self.iter_num = 0
        
        # margin_parameter
        self.beta = 0.25
        self.pair_m = 1.5
        self.d = 1

        self.encoder_q =EfficientNet()
        self.projection_q = nn.Sequential(
            nn.Linear(1792,1792),
            nn.BatchNorm1d(1792),
            nn.ReLU(True),
            nn.Linear(1792,128),
            # nn.BatchNorm1d(128),
        )
        self.local_projection_q = nn.Sequential(
            nn.Conv2d(1792,1792,1,1,0),
            nn.BatchNorm2d(1792),
            nn.ReLU(True),
            nn.Conv2d(1792,128,1,1,0),
            # nn.BatchNorm2d(128),
        )
        self.predictor = nn.Sequential(
            nn.Linear(128,1792),
            nn.BatchNorm1d(1792),
            nn.ReLU(True),
            nn.Linear(1792,128)
        )
        self.local_predictor_q = nn.Sequential(
            nn.Conv2d(128,1792,1,1,0),
            nn.BatchNorm2d(1792),
            nn.ReLU(True),
            nn.Conv2d(1792,128,1,1,0)
        )

        self.encoder_p = copy.deepcopy(self.encoder_q)
        self.projection_p = copy.deepcopy(self.projection_q)
        self.local_projection_p = copy.deepcopy(self.local_projection_q)

        for param_q, param_p in zip(self.encoder_q.parameters(), self.encoder_p.parameters()):
            param_p.data.copy_(param_q.data)  # initialize
            param_p.requires_grad = False  # not update by gradient


        for param_q, param_p in zip(self.projection_q.parameters(), self.projection_p.parameters()):
            param_p.data.copy_(param_q.data)  # initialize
            param_p.requires_grad = False  # not update by gradient


        for param_q, param_p in zip(self.local_projection_q.parameters(), self.local_projection_p.parameters()):
            param_p.data.copy_(param_q.data)  # initialize
            param_p.requires_grad = False  # not update by gradient

        self.register_buffer('positive',torch.zeros(1792))
        self.register_buffer('negtive',torch.zeros(1792))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_p in zip(self.encoder_q.parameters(), self.encoder_p.parameters()):
            param_p.data = param_p.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_p in zip(self.projection_q.parameters(), self.projection_p.parameters()):
            param_p.data = param_p.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_p in zip(self.local_projection_q.parameters(), self.local_projection_p.parameters()):
            param_p.data = param_p.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q):

        feature,logits= self.encoder_q(im_q)
        return logits

if __name__ == '__main__':
    checkpoint = torch.load('models_params_28.tar',map_location='cpu')
    model = SCL()
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded successfully! Start customizing your evaluation code!')

