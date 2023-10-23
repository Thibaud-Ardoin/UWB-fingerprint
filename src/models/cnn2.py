if __name__ == "__main__":
	import sys
	sys.path.append("src")

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import params


#  LIL Encoding NETWORK
class CNN2(nn.Module):

    def __init__(self, expender_multiplier=1, dropout_value=0):
        super(CNN2, self).__init__()
        self.embedding_size = params.expender_out
        self.expender_multiplier = expender_multiplier
        self.dropout_value = params.dropout
        self.dropout = nn.Dropout(self.dropout_value) 
        
        # Fur conv encoder
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=2)

        self.norm = nn.BatchNorm1d(64)

        self.flatten1 = nn.Flatten()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14*64, 64)
        self.fc2 = nn.Linear(64, params.latent_dimention)
        self.softmax = nn.Softmax()
        
        if params.use_extender:
            self.expenderFc1 = nn.Linear(params.latent_dimention, int(params.expender_out))
            self.expenderFc2 = nn.Linear(int(params.expender_out), int(params.expender_out))
            self.expenderFc3 = nn.Linear(int(params.expender_out), params.expender_out)
        
        
    def encoder(self, x): 
        x = x[:, None, :]        
        x = self.conv1(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.dropout(x)
#         x = nn.ReLU()(x)
        
#         x = self.norm(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
#         x = self.softmax(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def expender(self, x) :
        # Just two Fc layers with augmenting size
        x = F.relu(self.expenderFc1(x))
        x = self.dropout(x)
        x = F.relu(self.expenderFc2(x))
        x = self.dropout(x)
        x = self.expenderFc3(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        if params.use_extender:
            x = self.expender(x)
        return x


if __name__ == "__main__":
	model = CNN2()
	print(model)

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)
