if __name__ == "__main__":
	import sys
	sys.path.append("src")

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import params

from models.arcface import ArcFace

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
        
        if params.data_type == "complex":
            input_channel = 2
        else:
            input_channel = 1
        feature_sizes = [input_channel, params.conv_features1_nb, params.conv_features1_nb, params.conv_features2_nb, params.conv_features2_nb, params.conv_features2_nb, params.conv_features2_nb]
        kernel_sizes = [params.conv_kernel1_size, params.conv_kernel1_size, params.conv_kernel1_size, params.conv_kernel2_size, params.conv_kernel2_size, params.conv_kernel2_size]
        
        if params.input_type == "spectrogram":
            out_size = params.spectrogram_window_size
            out_size2 = params.spectrogram_window_size*(params.additional_samples+1)
            for i in range(params.conv_layers_nb):
        #            print(kernel_sizes[i])
                self.convs.append(nn.Conv2d(
                    feature_sizes[i],
                    feature_sizes[i+1], 
                    kernel_size=kernel_sizes[i], 
                    stride=params.stride_size, 
                    padding=params.padding_size))
                out_size = math.floor(((math.ceil((out_size + 2*params.padding_size - (kernel_sizes[i] - 1))/params.stride_size)) - params.pooling_kernel_size) / params.pooling_stride_size +1)
                out_size2 = math.floor(((math.ceil((out_size2 + 2*params.padding_size - (kernel_sizes[i] - 1))/params.stride_size)) - params.pooling_kernel_size) / params.pooling_stride_size +1)
                # without maxpool
                #out_size = math.ceil((out_size + 2*params.padding_size - (kernel_sizes[i] - 1))/params.stride_size)
        else:
            out_size = params.signal_length
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

        if params.input_type == "spectrogram":
            self.max_pool = nn.MaxPool2d(kernel_size=params.pooling_kernel_size, stride=params.pooling_stride_size)
        else:    
            self.max_pool = nn.MaxPool1d(kernel_size=params.pooling_kernel_size, stride=params.pooling_stride_size)

        if params.feature_norm == "batch":
            self.norm = nn.BatchNorm1d(feature_sizes[params.conv_layers_nb+1])
            if params.input_type == "spectrogram":
                self.norm = nn.BatchNorm2d(feature_sizes[params.conv_layers_nb+1])
        elif params.feature_norm == "layer":
            if params.input_type == "spectrogram":
                self.norm = nn.LayerNorm([out_size, out_size2])
            else: 
                self.norm = nn.LayerNorm(out_size)
        else:
            self.norm = nn.Identity()

        # TAIL FC
        self.flatten = nn.Flatten()
        self.fcs = []
        if params.input_type == "spectrogram":
            d1 = feature_sizes[params.conv_layers_nb+1] * out_size * out_size2
        else:
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

        self.arcface = ArcFace(params.latent_dimention, params.num_dev, params.scale, params.margin)
        # Expender
        exp_layer_sizes = [params.latent_dimention]
        exp_layer_sizes = exp_layer_sizes + [int(params.expender_hidden_size) for _ in range(params.expender_layers_nb -1 )]
        exp_layer_sizes.append(params.expender_out)
        self.expFcs = []
        for i in range(params.expender_layers_nb):
            self.expFcs.append( nn.Linear(exp_layer_sizes[i], exp_layer_sizes[i+1]))
        self.expFcs = nn.ModuleList(self.expFcs)

        # # POSITIONAL ENCODE
        self.fc_p1 = nn.Linear(4, 128)
        self.fc_p2 = nn.Linear(128, 1)


    def positional_encoder(self, x):
        # INPUT: (4, 250) with real, imag, cos, sin
        # x = x.sum(1)
        return x[:, 0]
        x = x.transpose(1, 2)
        x = F.relu(self.fc_p1(x))
        x = self.dropout(x)
        x = self.fc_p2(x).squeeze()
        return x


        
    def encoder(self, x):
        x = x[:, None, :]
        if params.data_type == "complex":
            # Reshape the input to have 2 channels
            x = x.view(x.shape[0], -1, x.shape[2])

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
    
    def classifier(self, x, label=None):
        if params.arcface == True:
            x = self.arcface(x, label)
        else:
            for i in range(params.class_layers_nb):
                x = self.clsFcs[i](x)
                x = self.dropout(x)
                if i < params.class_layers_nb - 1:
                    x = F.relu(x)
        if params.loss != "CrossentropyLoss":
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

    def classify(self, x, label=None):
        return self.classifier(x, label)

    def encode(self, x):
        if params.data_use_position:
            x = self.positional_encoder(x)
        return self.encoder(x)

    def expand(self, x):
        return self.expander(x)

    def forward(self, x, label=None):
        x = self.encode(x)
        if params.loss=="vicreg":
            x = self.expander(x)
        else:
            x = self.classify(x, label)
        return x


if __name__ == "__main__":
	model = ClassCNN1()
	print(model)

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)
