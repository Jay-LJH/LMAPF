import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from alg_parameters import *


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MAP_ACNet(nn.Module):
    def __init__(self):
        """initialization"""
        super(MAP_ACNet, self).__init__()
        gain = nn.init.calculate_gain('relu')

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

        def init2_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), CopParameters.GAIN)

        def init3_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.downsample1 = nn.Conv2d(CopParameters.OBS_CHANNEL, CopParameters.NET_SIZE// 2, kernel_size=1, stride=1, bias=False)
        self.conv1 = nn.Conv2d(CopParameters.OBS_CHANNEL,CopParameters.NET_SIZE // 2,kernel_size=3,stride=1,padding=1,groups=1,bias=False, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(CopParameters.NET_SIZE // 2,CopParameters.NET_SIZE // 2,kernel_size=3,stride=1,padding=1,groups=1,bias=False, dilation=1)
        self.maxpool = nn.MaxPool2d(2, 2) # pool window:2, stride:
        self.downsample2 = nn.Conv2d(CopParameters.NET_SIZE // 2, CopParameters.NET_SIZE, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(CopParameters.NET_SIZE // 2,CopParameters.NET_SIZE,kernel_size=3,stride=1,padding=1,groups=1,bias=False, dilation=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(CopParameters.NET_SIZE,CopParameters.NET_SIZE,kernel_size=3,stride=1,padding=1,groups=1,bias=False, dilation=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fully_connected_1 = init_(nn.Linear(runParameters.VEC_LEN, CopParameters.NET_VEC))
        self.fully_connected_2 = init_(nn.Linear(CopParameters.NET_SIZE, CopParameters.NET_SIZE))
        self.fully_connected_3 = init_(nn.Linear(CopParameters.NET_SIZE, CopParameters.NET_SIZE))
        self.lstm_memory = nn.LSTMCell(input_size=CopParameters.NET_SIZE, hidden_size=CopParameters.NET_SIZE)
        for name, param in self.lstm_memory.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # output heads
        self.policy_layer = init2_(nn.Linear(CopParameters.NET_SIZE, CopParameters.OUTPUT_ACTION))
        self.softmax_layer = nn.Softmax(dim=-1)
        self.value_layer = init3_(nn.Linear(CopParameters.NET_SIZE, 1))

        self.layer_norm_1 = nn.LayerNorm(CopParameters.NET_SIZE)
        self.layer_norm_2 =nn.LayerNorm(CopParameters.NET_SIZE)
        self.layer_norm_3 = nn.LayerNorm(CopParameters.NET_SIZE)
        self.layer_norm_4 = nn.LayerNorm(CopParameters.NET_SIZE)

    @autocast()
    def forward(self, x, hidden_state):
        """run neural network"""
        num_agent = x.shape[1]
        x = torch.reshape(x, (-1, CopParameters.OBS_CHANNEL, CopParameters.FOV, CopParameters.FOV))
        identity = self.downsample1(x)  
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += identity
        x = self.relu(x)
        x = self.maxpool(x)

        identity = self.downsample2(x)
        x = self.conv3(x)
        x = self.relu2(x)
        x = self.conv4(x)
        x += identity
        x = self.relu2(x) 

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.layer_norm_1(x)

        # x_1=F.relu(self.fully_connected_1(x_1))
        # x=torch.cat((x,x_1),-1)
        # x = self.layer_norm_2(x)
        
        # residual connection
        identity=x
        x = F.relu(self.fully_connected_2(x))
        x = self.fully_connected_3(x)
        x = F.relu(identity + x)
        x = self.layer_norm_3(x)

        x, memory_c = self.lstm_memory(x,hidden_state)
        hidden_state = (x, memory_c)
        x = torch.reshape(x, (-1, num_agent, CopParameters.NET_SIZE))
        x =self.layer_norm_4(x)
        policy_layer = self.policy_layer(x)
        policy = self.softmax_layer(policy_layer)
        policy_sig = torch.sigmoid(policy_layer)
        value = self.value_layer(x)
        return policy,value, policy_sig,hidden_state

if __name__ == '__main__':
    net=MAP_ACNet()
    obs = torch.torch.rand(
        (3,384, CopParameters.OBS_CHANNEL, CopParameters.FOV, CopParameters.FOV),
        dtype=torch.float32)
    vec = torch.torch.rand((3,384, runParameters.VEC_LEN), dtype=torch.float32)
    hidden_state = (
        torch.torch.rand((384 * 3, CopParameters.NET_SIZE)),
        torch.torch.rand((384* 3, CopParameters.NET_SIZE)))  # [B*A,3]
    #
    policy, value,policy_sig, output_state = net(obs,vec, hidden_state)
    print("test")
    print(f"policy shape: {policy.shape}\nvalue: {value.shape}")
