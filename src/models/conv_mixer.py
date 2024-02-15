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
				
		self.embedding_size = 128 				# "h" in ConvMix paper
		self.patch_size = 7 					# "p" in ConvMix paper
		self.depth = 5							# Number of ConvMixer layers
		self.kernel_size = 20 					# Size of depth conv's kernel
		self.out_size = params.num_dev			# Output size

		self.Seq = nn.Sequential																	# Sequence compiler
		self.ActBn = lambda x: self.Seq(x, nn.GELU(), nn.BatchNorm1d(self.embedding_size))			# Activation + Batch norm
		self.Residual = type('Residual', (self.Seq,), {'forward': lambda self, x: self[0](x) + x}) 	# Define Residual connection

		# Define 1st row of path making
		self.patch_maker = self.ActBn(nn.Conv1d(1, self.embedding_size, kernel_size=self.patch_size, stride=self.patch_size)) 
		
		self.conv_mixers = self.Seq(*[self.Seq(
								self.Residual(self.ActBn(nn.Conv1d(self.embedding_size, self.embedding_size, self.kernel_size, groups=self.embedding_size, padding="same"))), 
							   	self.ActBn(nn.Conv1d(self.embedding_size, self.embedding_size, 1))) for i in range(self.depth)])
							
		self.tail = self.Seq(nn.AdaptiveAvgPool1d((1)), nn.Flatten())
		
		self.classifier = nn.Linear(self.embedding_size, self.out_size)

	def encode(self, x):
		x = x.unsqueeze(1)
		x = self.patch_maker(x)
		x = self.conv_mixers(x)
		x = self.tail(x)
		return x
	
	def classify(self, x):
		# For output of alreaddy encoded x
		x = self.classifier(x)
		return x


	def forward(self, x):
		x = x.unsqueeze(1)
		x = self.patch_maker(x)

		x = self.conv_mixers(x)

		x = self.tail(x)

		x = self.classifier(x)	

		return x



if __name__ == "__main__":
	
	model = ConvMixer()

	bsz = 256
	sign_length = 200

	dummy_data = torch.rand((bsz, sign_length))
	out = model(dummy_data)
	print(out.shape)