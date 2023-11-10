if __name__ == "__main__":
	import sys
	sys.path.append("src")

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import params
import models


#  LIL Encoding NETWORK
class Transformer3(nn.Module):
	"""
		Slice the signal into different bins of values that are used as embedded values
	"""
	
	def __init__(self):
		super(Transformer3, self).__init__()
		self.use_extender = params.use_extender
		self.dropout_value = params.dropout_value
		self.dropout = nn.Dropout(self.dropout_value)
		self.embedding_size = params.trans_embedding_size * params.trans_head_nb

		# ATTENTION
		self.trans_embedding_size = params.trans_embedding_size * params.trans_head_nb # Divisibility needed
		self.trans_head_nb = params.trans_head_nb
		self.trans_layer_nb = params.trans_layer_nb
		self.trans_hidden_nb = params.trans_hidden_nb
		self.flatten = nn.Flatten()

		self.window_size = params.window_size
		self.window_nb = math.ceil(params.signal_length/self.window_size)

		encoder_layers = nn.TransformerEncoderLayer(self.embedding_size, self.trans_head_nb, self.trans_hidden_nb, self.dropout_value, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.trans_layer_nb)

		# PRE-PROCESS
		self.fc0 = nn.Linear(self.window_size, self.embedding_size)

		# MIDLE WORK
		self.norm = nn.LayerNorm((self.window_nb, self.embedding_size))
		self.flatten = nn.Flatten()

		# MLP
		self.fc1 = nn.Linear(self.embedding_size * self.window_nb, params.latent_dimention*2)
		self.fc2 = nn.Linear(params.latent_dimention*2, params.latent_dimention)
		self.softmax = nn.Softmax()
		
		# EXPENDER
		self.batchnorm1 = nn.BatchNorm1d(params.latent_dimention)
		self.batchnorm2 = nn.BatchNorm1d(params.latent_dimention*4)
		self.expenderFc1 = nn.Linear(params.latent_dimention, params.latent_dimention*4)
		# self.expenderFc2 = nn.Linear(params.latent_dimention + int(params.expender_out/4), params.latent_dimention + int(params.expender_out/2))
		self.expenderFc2 = nn.Linear(params.latent_dimention*4, params.latent_dimention*4)
		# self.expenderFc3 = nn.Linear(params.latent_dimention + int(params.expender_out/2), params.expender_out)
		self.expenderFc3 = nn.Linear(params.latent_dimention*4, params.latent_dimention*4)

	def preprocess(self, x):
		# Can be double-1 so we overlap half of it, 10 values
		div = self.window_nb #math.ceil(200/self.window_size)
		total_overlap = (params.signal_length - div * self.window_size)
		minimal_side_overlap = round(total_overlap/(div*2))

		l = []
		ind = 0
		for i in range(div):
			if ind+self.window_size > params.signal_length:
				end = params.signal_length
				ind = params.signal_length - self.window_size
			else :
				end = ind + self.window_size
			
			l.append(x[:, ind:end])
			ind = end

		x = torch.stack(l, dim=1)
		# x = torch.reshape(x, (x.size(0), self.window_nb, self.window_size))

		x = self.fc0(x)

		return x


	def encoder(self, x): 
		x = self.preprocess(x)

		x = self.transformer_encoder(x)
		# x = self.norm(x)

		x = self.flatten(x)		

		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)

		x = F.normalize(x, p=2, dim=1)
		
		return x
	
	def expender(self, x) :
		# Just two Fc layers with augmenting size
		x = F.relu(self.expenderFc1(self.batchnorm1(x)))
		x = self.dropout(x)

		# x = self.batchnorm2(F.relu(self.expenderFc2(x)))
		# x = self.dropout(x)

		x = self.expenderFc3(x)
		return x
	
	def forward(self, x):
		x = self.encoder(x)
		if self.use_extender:
			x = self.expender(x)
		return x
	

if __name__ == "__main__":
	
	model = Transformer3()

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)