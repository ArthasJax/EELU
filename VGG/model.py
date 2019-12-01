import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import activations.activations as AF
import math


#act_parm : **{'act_name':'EELU', 'alpha':1, 'beta':1, 'eps':0.6, 'mode':'channel-shared' or 'channel-wise'}
def activation_function(out_channel, act_parm):
    if act_parm['act_name'] == 'relu':
        activation = nn.ReLU(inplace=True)
    elif act_parm['act_name'] == 'MPELU':
        if act_parm['mode'] == 'channel-shared':
            activation = AF.MPELU(num_parameters=1, pa_init=act_parm['alpha'], pb_init=act_parm['beta'])
        else:
            activation = AF.MPELU(num_parameters=out_channel, pa_init=act_parm['alpha'], pb_init=act_parm['beta'])
    elif act_parm['act_name'] == 'EPReLU':
        if act_parm['mode'] == 'channel-shared':
            activation = AF.EPReLU(num_parameters=1, eps=act_parm['eps'])
        else:
            activation = AF.EPReLU(num_parameters=out_channel, eps=act_parm['eps']) 
    elif act_parm['act_name'] == 'EELU':
        if act_parm['mode'] == 'channel-shared':
            activation = AF.EELU(num_parameters=1, pa_init=act_parm['alpha'], pb_init=act_parm['beta'], eps=act_parm['eps'])
        else:
            activation = AF.EELU(num_parameters=out_channel, pa_init=act_parm['alpha'], pb_init=act_parm['beta'], eps=act_parm['eps'])
    elif act_parm['act_name'] == 'EReLU':
        activation = AF.EReLU(eps=act_parm['eps'])
    elif act_parm['act_name'] == 'ELU':
        activation = nn.ELU(inplace=True)
    elif act_parm['act_name'] == 'Swish':
        activation = AF.Swish(inplace=True)
    return activation


class VGG_16(nn.Module):
    def __init__(self, input_size=224, input_channel=3, num_classes=1000, act_parm={}):
        super(VGG_16, self).__init__()
        
        self.input_size = input_size
        self.input_channel = input_channel
        
        if act_parm['act_name'] not in ('relu', 'MPELU', 'EELU', 'EReLU', 'ELU', 'EPReLU', 'Swish'):
            raise ValueError("\"{}\" Activation Name error".format(act_parm['act_name']))
        
        
        self.Conv_1 = nn.Conv2d(self.input_channel, out_channels=64, kernel_size=3, padding=1)
        self.Batch_Norm_1 = nn.BatchNorm2d(64)
        
        self.act_1 = activation_function(64, act_parm)
        
        self.Conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.Batch_Norm_2 = nn.BatchNorm2d(64)
        
        self.act_2 = activation_function(64, act_parm)
        
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        
        self.Conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.Batch_Norm_3 = nn.BatchNorm2d(128)
            
        self.act_3 = activation_function(128, act_parm)
        
        self.Conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.Batch_Norm_4 = nn.BatchNorm2d(128)
            
        self.act_4 = activation_function(128, act_parm)
        
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        
        self.Conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.Batch_Norm_5 = nn.BatchNorm2d(256)
            
        self.act_5 = activation_function(256, act_parm)
        
        self.Conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.Batch_Norm_6 = nn.BatchNorm2d(256)
            
        self.act_6 = activation_function(256, act_parm)
        
        self.Conv_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.Batch_Norm_7 = nn.BatchNorm2d(256)
            
        self.act_7 = activation_function(256, act_parm)
        
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2)
        
        self.Conv_8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.Batch_Norm_8 = nn.BatchNorm2d(512)
            
        self.act_8 = activation_function(512, act_parm)
        
        self.Conv_9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.Batch_Norm_9 = nn.BatchNorm2d(512)
            
        self.act_9 = activation_function(512, act_parm)
        
        self.Conv_10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.Batch_Norm_10 = nn.BatchNorm2d(512)
            
        self.act_10 = activation_function(512, act_parm)
        
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2)
        
        self.Conv_11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.Batch_Norm_11 = nn.BatchNorm2d(512)
        
        self.act_11 = activation_function(512, act_parm)
        
        self.Conv_12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.Batch_Norm_12 = nn.BatchNorm2d(512)
            
        self.act_12 = activation_function(512, act_parm)
        
        self.Conv_13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.Batch_Norm_13 = nn.BatchNorm2d(512)
            
        self.act_13 = activation_function(512, act_parm)
        
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2)
        
        self.fc_1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.act_14 = activation_function(1, act_parm)
        
        self.fc_2 = nn.Linear(in_features=4096, out_features=4096)
        self.act_15 = activation_function(1, act_parm)
        
        self.fc_3 = nn.Linear(in_features=4096, out_features=num_classes)
                
        #self.softmax = nn.Softmax()
        
        ########## Initialization #########
        for name, param in self.named_parameters():
            if 'Norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'Norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'Conv' in name and 'weight' in name:
                n = (param.size(1) * param.size(2) * param.size(3)) * (1 + act_parm['alpha']**2 * act_parm['beta']**2)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'pa' in name:
                param.data.fill_(act_parm['alpha'])
            elif 'pb' in name:
                param.data.fill_(act_parm['beta'])
            elif 'fc_':
                param.data.normal_().mul_(0.01)
                
            
        
    def forward(self, input):
        input = self.Conv_1(input)
        input = self.Batch_Norm_1(input)
        input = self.act_1(input)
        
        input = self.Conv_2(input)
        input = self.Batch_Norm_2(input)
        input = self.act_2(input)
        
        input = self.maxpool_1(input)
        
        input = self.Conv_3(input)
        input = self.Batch_Norm_3(input)
        input = self.act_3(input)
        
        input = self.Conv_4(input)
        input = self.Batch_Norm_4(input)
        input = self.act_4(input)
    
        input = self.maxpool_2(input)
        
        input = self.Conv_5(input)
        input = self.Batch_Norm_5(input)
        input = self.act_5(input)
        
        input = self.Conv_6(input)
        input = self.Batch_Norm_6(input)
        input = self.act_6(input)
        
        input = self.Conv_7(input)
        input = self.Batch_Norm_7(input)
        input = self.act_7(input)

        input = self.maxpool_3(input)
                
        
        input = self.Conv_8(input)
        input = self.Batch_Norm_8(input)
        input = self.act_8(input)
        
        input = self.Conv_9(input)
        input = self.Batch_Norm_9(input)
        input = self.act_9(input)
        
        input = self.Conv_10(input)
        input = self.Batch_Norm_10(input)
        input = self.act_10(input)
        
        input = self.maxpool_4(input)
        
        input = self.Conv_11(input)
        input = self.Batch_Norm_11(input)
        input = self.act_11(input)
        
        input = self.Conv_12(input)
        input = self.Batch_Norm_12(input)
        input = self.act_12(input)
        
        input = self.Conv_13(input)
        input = self.Batch_Norm_13(input)
        input = self.act_13(input)
        
        input = self.maxpool_5(input)
        
        input = torch.flatten(input, 1)
       
        input = F.dropout(self.act_14(self.fc_1(input)))
        input = F.dropout(self.act_15(self.fc_2(input)))
        logit = self.fc_3(input)
        
        return logit
    
    

        
if __name__ == "__main__":
    """
    testing
    """
    #act_parm : **{'act_name':'EELU', 'alpha':1, 'beta':1, 'eps':0.6, 'mode':'channel-shared' or 'channel-wise'}
    #act_parm = {'act_name':'EELU', 'alpha':0, 'beta':0, 'eps':1.0, 'mode':'channel-wise'}
    act_parm = {'act_name': 'EPReLU', 'alpha': 0, 'beta': 0, 'eps': 0.4, 'mode': 'channel-wise'}
    #act_parm = {'act_name': 'EReLU', 'alpha': 0, 'beta': 0, 'eps': 0.4, 'mode': 'channel-wise'}
    
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")   
    
    model = VGG_16(act_parm = act_parm)
    model = model.cuda()
    
    x = torch.cuda.FloatTensor(np.random.random((2, 3, 224, 224)))
    #out = model(x).to(device)
    for _ in range(2):
        out = model(x)
        loss = torch.sum(out)
        loss.backward()      
        
        print(loss)