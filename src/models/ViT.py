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
from models.arcface import ArcFace

class EmbeddingPatches(nn.Module):
    """Turns a 1d or 2d input into a 1D learnable vector embedding.
    
    Args:
        input_channels (int): Number of channels for the input.
        size_of_patch (int): Size of patches to convert the input.
        embedding_size (int): Size of embedding output.
    """ 
    def __init__(self, 
                 input_channels:int=1,
                 size_of_patch:int=params.window_size,
                 embedding_size:int=params.trans_embedding_size):
        super().__init__()
        
        self.size_of_patch = size_of_patch
        
        if params.input_type == "spectrogram":

            self.patch = nn.Conv2d(in_channels=input_channels,
										out_channels=embedding_size,
										kernel_size=size_of_patch,
										stride=size_of_patch,
										padding=0)
        else:
            self.patch = nn.Conv1d(in_channels=input_channels,
									out_channels=embedding_size,
									kernel_size=size_of_patch,
									stride=size_of_patch,
									padding=0)
    
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
    
    def forward(self, x):

        input_size = x.shape[-1]
        assert input_size % self.size_of_patch == 0, "The input must be divisible by the size of the patch! input_size:{:5f} self.size_of_patch:{:5f}".format(input_size, self.size_of_patch)

        x = self.patch(x)
        if params.input_type == "spectrogram":
            
            flat_x = self.flatten(x) 

            return flat_x.permute(0, 2, 1)
        else:
            return x.permute(0, 2, 1)



class ViT(nn.Module):
	"""
		Slice the signal into patches that are used as embedded values
	"""
	
	def __init__(self):
		super(ViT, self).__init__()
		self.dropout_value = params.dropout_value
		self.dropout = nn.Dropout(self.dropout_value)
		self.embedding_size = params.trans_embedding_size * params.trans_head_nb
		self.signal_length = params.signal_length
		self.num_dev = params.num_dev
		self.input_type = params.input_type
		self.data_type = params.data_type
		self.latent_dimention = params.latent_dimention
		self.window_size = params.window_size
	
		# ATTENTION
		self.trans_embedding_size = params.trans_embedding_size * params.trans_head_nb # Divisibility needed
		self.trans_head_nb = params.trans_head_nb
		self.trans_layer_nb = params.trans_layer_nb
		self.trans_hidden_nb = params.trans_hidden_nb

		self.flatten = nn.Flatten()
	
		if self.input_type == "spectrogram":
			self.window_nb = math.ceil((params.spectrogram_window_size*(params.spectrogram_window_size*(params.additional_samples+1)))/(self.window_size**2)) # adjust parameters
		else:
			self.window_nb = math.ceil(self.signal_length/self.window_size)
	
		encoder_layers = nn.TransformerEncoderLayer(self.embedding_size, self.trans_head_nb, self.trans_hidden_nb, self.dropout_value, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.trans_layer_nb)
	
		# MLP head for classification (simple ViT)
		self.mlp_head = nn.Sequential(
			nn.LayerNorm(normalized_shape=self.embedding_size),
			nn.Linear(in_features=self.embedding_size,
					out_features=self.num_dev)
		)

		# CLS Token
		self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_size),
										requires_grad=True)
	
		# Pos embedding
		count_patches = self.window_nb
		self.pos_embedding = nn.Parameter(torch.randn(1, count_patches, self.embedding_size))
	
		# MLP
		#self.fc1 = nn.Linear((self.embedding_size * self.window_nb), self.latent_dimention*2)

		#self.fc2 = nn.Linear(self.latent_dimention*2, self.latent_dimention)
		self.class_fc = nn.Linear(self.embedding_size, self.num_dev)
		self.softmax = nn.Softmax(dim=1)

		if self.data_type == "complex":
			channels = 2
		else:
			channels = 1
		self.patch_embedding = EmbeddingPatches(input_channels=channels,
	                                      size_of_patch=self.window_size,
	                                      embedding_size=self.embedding_size)

		self.arcface = ArcFace(self.embedding_size, self.num_dev, params.arcface_scale, params.arcface_margin)

	def preprocess(self, x):
		x = x[:, None, :]
		if self.data_type == "complex":
			# Reshape the input to have 2 channels
			x = x.view(x.shape[0], -1, x.shape[2])
		# Get some dimensions from x
		#batch_size = x.shape[0]
		x = self.patch_embedding(x)

		# x = x.transpose(1, 2)# expand the class token across the batch size
		#cls_token = self.cls_token.expand(batch_size, -1, -1) # "-1" to infer the dimension
		# add class token to the patch embedding
		#x = torch.cat((cls_token, x), dim=1)

		# Add the positional embedding to patch embedding (and the cls token if used)
		x = self.pos_embedding + x
		

		return x


	def encode(self, x): 
		x = self.preprocess(x)

		x = self.transformer_encoder(x)
		x = x.mean(dim = 1)
		x = F.normalize(x, p=2, dim=1)
		# x = self.norm(x)

		#x = self.flatten(x)		

		#x = F.relu(self.fc1(x))
		#x = self.dropout(x)
		#x = self.fc2(x)
		#x = F.normalize(x, p=2, dim=1)
		
		return x
	
	def expand(self, x):
		return x
	
	def classify(self, x, label=None):
		# Pass 0th index of x through MLP head
    	#x = self.mlp_head(x[:, 0]) # for cls token
		#x = self.mlp_head(x)
		if params.arcface == True:
			x = self.arcface(x, label)
		else:
			x = self.class_fc(x)
		if params.loss != "CrossentropyLoss":
			x = self.softmax(x)
		return x		
	
	def forward(self, x, label=None):
		x = self.encode(x)
		x = self.classify(x, label)
		return x


if __name__ == "__main__":
	
	model = ViT()
	
	bsz = 256
	signal_length = 250
	
	dummy_data = torch.rand((bsz, signal_length))
	out = model(dummy_data)
	print(out.shape)


if __name__ == "__main__":

  import sys

  sys.path.append("src")