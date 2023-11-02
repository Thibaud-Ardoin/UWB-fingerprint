if __name__ == "__main__":
	import sys
	sys.path.append("src")

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import params


#  LIL Encoding NETWORK
class ClassCNN1(nn.Module):

    def __init__(self, expender_multiplier=1, dropout_value=0):
        super(ClassCNN1, self).__init__()
        self.embedding_size = params.expender_out
        self.expender_multiplier = expender_multiplier
        self.dropout_value = params.dropout_value
        self.dropout = nn.Dropout(self.dropout_value)
        
        # Fur conv encoder
        self.convs = []
        out_size = params.signal_length
        feature_sizes = [1, params.conv_features1_nb, params.conv_features1_nb, params.conv_features2_nb, params.conv_features2_nb, params.conv_features2_nb, params.conv_features2_nb]
        kernel_sizes = [params.conv_kernel1_size, params.conv_kernel1_size, params.conv_kernel1_size, params.conv_kernel2_size, params.conv_kernel2_size, params.conv_kernel2_size]
        for i in range(params.conv_layers_nb):
#            print(kernel_sizes[i])
            self.convs.append(nn.Conv1d(
                feature_sizes[i],
                feature_sizes[i+1], 
                kernel_size=kernel_sizes[i], 
                stride=params.stride_size, 
                padding=params.padding_size))
            out_size = math.ceil((out_size + 2*params.padding_size - (kernel_sizes[i] - 1))/params.stride_size)
#            print(out_size)
        self.convs = nn.ModuleList(self.convs)

        # self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2)
        # self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        # self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=2)

        if params.feature_norm == "batch":
            self.norm = nn.BatchNorm1d(feature_sizes[params.conv_layers_nb+1])
        elif params.feature_norm == "layer":
            self.norm = nn.LayerNorm(out_size)
        else:
            self.norm = nn.Identity()

        # TAIL FC
        self.flatten = nn.Flatten()
        self.fcs = []
        d1 = feature_sizes[params.conv_layers_nb+1] * out_size
#feature_sizes[params.conv_layers_nb+1] * math.ceil(params.signal_length / ((params.stride_size)**params.conv_layers_nb))
 #       print(d1)
        dim_size = [d1, params.latent_dimention, params.latent_dimention, params.latent_dimention]
        if params.tail_fc_layers_nb>1:
             dim_size[1] = dim_size[1]*2
        # 14*64
        for i in range(params.tail_fc_layers_nb):
            self.fcs.append( nn.Linear(dim_size[i], dim_size[i+1]) )

        self.fcs = nn.ModuleList(self.fcs)

        # self.fc2 = nn.Linear(64, params.latent_dimention)
        self.softmax = nn.Softmax(dim=1)

        # CLS
        self.clsFc1 = nn.Linear(params.latent_dimention, int(params.expender_out))
        self.clsFc2 = nn.Linear(int(params.expender_out), int(params.expender_out))
        self.clsFc3 = nn.Linear(int(params.expender_out), params.num_dev)
        
        
    def encoder(self, x): 
        x = x[:, None, :]

        # Conv layers
        for i in range(params.conv_layers_nb):
            x = self.convs[i](x)
            x = self.dropout(x)
            if i < params.conv_layers_nb -1:
                x = nn.ReLU()(x)
        x = self.norm(x)
        x = self.flatten(x)


        # Tail fc
        for i in range(params.tail_fc_layers_nb):
            x = self.fcs[i](x)
            x = self.dropout(x)
            x = F.relu(x)
    
        # x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)        
        return x
    
    def classifier(self, x) :
        # Just two Fc layers with augmenting size
        x = F.relu(self.clsFc1(x))
        x = self.dropout(x)
        x = F.relu(self.clsFc2(x))
        x = self.dropout(x)
        x = self.clsFc3(x)
        x = self.softmax(x)
        return x

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
	model = ClassCNN1()
	print(model)

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)