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
        out_size = (params.additional_samples+1)*200
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
            out_size = math.floor(((math.ceil((out_size + 2*params.padding_size - (kernel_sizes[i] - 1))/params.stride_size)) - params.pooling_kernel_size) / params.pooling_stride_size +1)
            # without maxpool
            #out_size = math.ceil((out_size + 2*params.padding_size - (kernel_sizes[i] - 1))/params.stride_size)
        self.convs = nn.ModuleList(self.convs)
        self.max_pool = nn.MaxPool1d(kernel_size=params.pooling_kernel_size, stride=params.pooling_stride_size)

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
        cls_layer_sizes = [params.latent_dimention]
        cls_layer_sizes = cls_layer_sizes + [int(params.class_hidden_size) for _ in range(params.class_layers_nb -1 )]
        cls_layer_sizes.append(params.num_dev)
        self.clsFcs = []
        for i in range(params.class_layers_nb):
            self.clsFcs.append( nn.Linear(cls_layer_sizes[i], cls_layer_sizes[i+1]) )
        self.clsFcs = nn.ModuleList(self.clsFcs)

        # Expender
        exp_layer_sizes = [params.latent_dimention]
        exp_layer_sizes = exp_layer_sizes + [int(params.expender_hidden_size) for _ in range(params.expender_layers_nb -1 )]
        exp_layer_sizes.append(params.expender_out)
        self.expFcs = []
        for i in range(params.expender_layers_nb):
            self.expFcs.append( nn.Linear(exp_layer_sizes[i], exp_layer_sizes[i+1]))
        self.expFcs = nn.ModuleList(self.expFcs)
        
    def encoder(self, x): 
        x = x[:, None, :]

        # Conv layers
        for i in range(params.conv_layers_nb):
            x = self.convs[i](x)
            x = self.max_pool(x)
            x = self.dropout(x)
            if i < params.conv_layers_nb -1:
                x = F.relu(x)
        x = self.norm(x)
        x = self.flatten(x)


        # Tail fc
        for i in range(params.tail_fc_layers_nb):
            x = self.fcs[i](x)
            x = self.dropout(x)
            if i < params.tail_fc_layers_nb -1:
                x = F.relu(x)
    
        x = F.normalize(x, p=2, dim=1)        
        return x
    
    def classifier(self, x) :
        for i in range(params.class_layers_nb):
            x = self.clsFcs[i](x)
            x = self.dropout(x)
            if i < params.class_layers_nb - 1:
                x = F.relu(x)
    
        x = self.softmax(x)
        return x

    def expander(self, x) :
        if params.use_extender and params.expender_layers_nb > 0:
            for i in range(params.expender_layers_nb):
                x = self.expFcs[i](x)
                x = self.dropout(x)
                if i < params.expender_layers_nb - 1:
                    x = F.relu(x)
        
            x = self.softmax(x)
        return x

    def classify(self, x):
        return self.classifier(x)

    def encode(self, x):
        return self.encoder(x)

    def expand(self, x):
        return self.expander(x)

    def forward(self, x):
        x = self.encoder(x)
        if params.loss=="vicreg":
            x = self.expander(x)
        else:
            x = self.classify(x)
        return x


if __name__ == "__main__":
	model = ClassCNN1()
	print(model)

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)
