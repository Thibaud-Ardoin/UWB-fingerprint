if __name__ == "__main__":
	import sys
	sys.path.append("src")

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import params


#  LIL Encoding NETWORK
class advCNN1(nn.Module):

    def __init__(self, expender_multiplier=1, dropout_value=0):
        super(advCNN1, self).__init__()
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
        self.fc1 = nn.Linear(14*64, params.latent_dimention*2)
        self.fc2 = nn.Linear(params.latent_dimention*2, params.latent_dimention)
        
        self.posFc1 = nn.Linear(params.latent_dimention, int(params.expender_out))
        self.posFc2 = nn.Linear(int(params.expender_out), int(params.num_pos))
        # self.posFc3 = nn.Linear(int(params.expender_out), params.expender_out)

        self.devFc1 = nn.Linear(params.latent_dimention, int(params.expender_out))
        self.devFc2 = nn.Linear(int(params.expender_out), int(params.num_dev))

        self.softmax = nn.Softmax()

        
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
        
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def posCLS(self, x) :
        # Classifier to determine the position label
        x = F.relu(self.posFc1(x))
        x = self.dropout(x)
        x = self.posFc2(x)
        x = self.softmax(x)
        return x

    def devCLS(self, x) :
        # Classifier to determine the device label
        x = F.relu(self.devFc1(x))
        x = self.dropout(x)
        x = self.devFc2(x)
        x = self.softmax(x)
        return x


    def forward(self, x):
        x = self.encoder(x)

        dev_pred = self.devCLS(x)

        pos_pred = self.posCLS(x)

        return dev_pred, pos_pred


if __name__ == "__main__":
	model = advCNN1()
	print(model)

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)
