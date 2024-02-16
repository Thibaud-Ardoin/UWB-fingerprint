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
class ConvMixer(nn.Module):
	def __init__(self):
		super(ConvMixer, self).__init__()

		self.embedding_size = params.convm_embedding_size 				# "h" in ConvMix paper
		self.patch_size = params.convm_patch_size 						# "p" in ConvMix paper
		self.depth = params.convm_layer_nb								# Number of ConvMixer layers
		self.kernel_size = params.convm_kernel_size 					# Size of depth conv's kernel
		self.out_size = params.convm_out_size							# Output size
		self.feature_size = params.signal_length//self.patch_size		# The feature size inside the convmixer

		self.Seq = nn.Sequential																	# Sequence compiler
		self.ActBn = lambda x: self.Seq(x, nn.GELU(), nn.BatchNorm1d(self.embedding_size))			# Activation + Batch norm
		self.Residual = type('Residual', (self.Seq,), {'forward': lambda self, x: self[0](x) + x}) 	# Define Residual connection

		# Define 1st row of path making
		self.patch_maker = self.ActBn(nn.Conv1d(1, self.embedding_size, kernel_size=self.patch_size, stride=self.patch_size)) 
		
		self.conv_mixers = self.Seq(*[self.Seq(
								self.Residual(self.ActBn(nn.Conv1d(self.embedding_size, self.embedding_size, self.kernel_size, groups=self.embedding_size, padding="same"))), 
							   	self.ActBn(nn.Conv1d(self.embedding_size, self.embedding_size, 1))) for i in range(self.depth)])
							
		# self.tail = nn.Flatten()
		self.tail = self.Seq(nn.AdaptiveAvgPool1d((1)), nn.Flatten())
		
		self.expander = nn.Linear(self.embedding_size, self.out_size)
		self.classifier = nn.Linear(self.embedding_size, params.num_dev)
		# self.classifier = nn.Linear(self.embedding_size*self.feature_size, self.out_size)

	def encode(self, x):
		x = x.unsqueeze(1)
		x = self.patch_maker(x)
		x = self.conv_mixers(x)
		x = self.tail(x)
		x = F.normalize(x, p=2, dim=1)

		return x
	
	def expand(self, x):
		x = self.expander (x)
		return x
	
	def classify(self, x):
		# For output of alreaddy encoded x
		x = self.classifier(x)
		return x

	def forward(self, x):
		x = self.encode(x)
		x = self.classifier(x)
		return x



if __name__ == "__main__":
	
	model = ConvMixer()

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)