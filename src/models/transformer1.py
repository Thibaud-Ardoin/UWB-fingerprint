import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import params
import models


#  LIL Encoding NETWORK
class Transformer1(nn.Module):
    """
        Ecnoding of the input made as a individual values on 200 different encoded time steps
    """

    def __init__(self, expender_multiplier=1, dropout_value=0):
        super(Transformer1, self).__init__()
        self.expender_multiplier = expender_multiplier
        self.use_extender = params.use_extender
        self.dropout_value = dropout_value
        self.dropout = nn.Dropout(self.dropout_value) 
        self.embed_size = params.expender_out
        
        # ATTENTION
        self.trans_embedding_size = params.trans_embedding_size * params.trans_head_nb # Divisibility needed
        self.trans_head_nb = params.trans_head_nb
        self.trans_layer_nb = params.trans_layer_nb
        self.trans_hidden_nb = params.trans_hidden_nb
        self.flatten = nn.Flatten()
        
        self.positions = torch.arange(200)
        
        
        self.trans_embeddings = nn.Linear(2, self.trans_embedding_size)
        self.pos_encoder = models.PositionalEncoding(self.trans_embedding_size, self.dropout_value)
        encoder_layers = nn.TransformerEncoderLayer(self.trans_embedding_size, self.trans_head_nb, self.trans_hidden_nb, self.dropout_value, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.trans_layer_nb)
#         self.embedding = nn.Linear(2, d_model)
        
        
        # CNN
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
#         self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2)
#         self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
#         self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=2)

        # MIDLE WORK
#         self.norm = nn.BatchNorm1d(32)
        self.norm = nn.LayerNorm((200, self.trans_embedding_size))
#         self.flatten1 = nn.Flatten()
        self.flatten = nn.Flatten()
#         self.fc1bis = nn.Linear(25*64, 64)
#         self.fc1bisbis = nn.Linear(200, 64)

        # MLP
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, params.latent_dimention)
        self.softmax = nn.Softmax()
        
        # EXPENDER
        self.expenderFc1 = nn.Linear(params.latent_dimention, params.latent_dimention + int(params.expender_out/4))
        self.expenderFc2 = nn.Linear(params.latent_dimention + int(params.expender_out/4), params.latent_dimention + int(params.expender_out/2))
        self.expenderFc3 = nn.Linear(int(params.expender_out/2), params.expender_out)


    def encoder(self, x): 
#         cx = x[:, None, :]        
#         cx = self.conv1(cx)
#         cx = self.dropout(cx)
#         cx = nn.ReLU()(cx)
#         cx = self.conv2(cx)
#         cx = self.dropout(cx)
#         cx = nn.ReLU()(cx)
#         cx = self.conv3(cx)
#         cx = self.dropout(cx)
#         cx = nn.ReLU()(cx)
        
#         cx = self.norm(cx)
#         x = x[:, None, :]        
#         print(x.shape)
#         x = torch.stack([x, self.positions[None].expand(x.shape)], dim=-1)
        x = x.unsqueeze(-1)
        x = self.trans_embeddings(torch.cat([x,x], dim=-1))    # Inputting 2 time the same data to create an random embedding 
    
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # summing the last trensformers dimention to narrrow it down
        x = self.norm(x)

        x = torch.sum(x, dim=-1)

        
#         x = self.flatten(x)
        
#         cx = self.fc1bis(cx)
#         x = self.fc1bisbis(x)            
#         x = x + cx
        
        
#         x = self.conv4(x)
#         x = self.dropout(x)
#         x = nn.ReLU()(x)
        
        
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
        if self.use_extender:
            x = self.expender(x)
        return x
    

if __name__ == "__main__":
    model = Transformer1()
    print(model)