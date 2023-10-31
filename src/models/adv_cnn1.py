if __name__ == "__main__":
	import sys
	sys.path.append("src")

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import params

class advEncoder(nn.Module):
    def __init__(self):
        super(advEncoder, self).__init__()
        self.dropout = nn.Dropout(params.dropout_value)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=2)

        self.norm = nn.BatchNorm1d(64)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14*64, params.latent_dimention*2)
        self.fc2 = nn.Linear(params.latent_dimention*2, params.latent_dimention)

    def forward(self, x):
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


class advCls(nn.Module):
    def __init__(self, output_size):
        super(advCls, self).__init__()
        self.dropout = nn.Dropout(params.dropout_value)
        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(params.latent_dimention, int(params.expender_out))
        self.fc2 = nn.Linear(int(params.expender_out), int(output_size))

    def forward(self, x) :
        # Classifier to determine the position label
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class advCNN1():

    def __init__(self, expender_multiplier=1, dropout_value=0):
        # super(advCNN1, self).__init__()

        self.encoder = advEncoder()
        self.posCls = advCls(params.num_pos)
        self.devCls = advCls(params.num_dev)

    def __call__(self, x):
        x = self.encoder(x)

        dev_pred = self.devCls(x)

        pos_pred = self.posCls(x)

        return dev_pred, pos_pred

    def classify(self, x):
        return self.devCls(x)
    

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.posCls = self.posCls.to(device)
        self.devCls = self.devCls.to(device)
        return self

    def train(self):
        # super(advCNN1, self).train()
        self.encoder.train()
        self.posCls.train()
        self.devCls.train()

    def eval(self):
        # super(advCNN1, self).eval()
        self.encoder.eval()
        self.posCls.eval()
        self.devCls.eval()


if __name__ == "__main__":
	model = advCNN1()
	print(model)

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)
