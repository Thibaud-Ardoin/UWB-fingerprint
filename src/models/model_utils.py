"""
	Model.py
	Links and agregate all the diferent model types and they specific architectures
"""
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

import params
import models


class PositionalEncoding(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.d_model = d_model
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		Arguments:
			x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
		"""
		p = self.pe[:x.size(1)][None].expand(list(x.shape))
		x = x + p
		return self.dropout(x)


def load_model():
	model = eval("models." + params.model_name)().to(params.device)
	if params.verbose:
		print(model)
		print("Total number of model's parameters", sum(p.numel() for p in model.parameters()))
	return model


if __name__ == "__main__":
	model = load_model()
	print(model)
