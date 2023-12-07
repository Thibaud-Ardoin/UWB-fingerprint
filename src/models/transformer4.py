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
class Transformer4(nn.Module):
	"""
		Slice the signal into different bins of values that are used as embedded values
	"""
	
	def __init__(self):
		super(Transformer4, self).__init__()
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

	def encode(self, x):
		return self.encoder(x)		

	def expand(self, x):
		return self.expander(x)	

	def classify(self, x):
		return self.classifier(x)		
	
	def forward(self, x):
		x = self.encoder(x)
		if self.use_extender:
			x = self.expand(x)
		return x
	

if __name__ == "__main__":
	
	model = Transformer4()

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)