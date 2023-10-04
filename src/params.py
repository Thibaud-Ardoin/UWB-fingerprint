"""
    Modifyable parameters file.
    Shoud use a synthax that allows grid search parametrisation too
"""
import pprint

import torch


##########
#   Data
##########
datafile = "/srv/public/Thibaud/datasets/ultrasec/Messung8/messung8.2_data.npy"
labelfile = "/srv/public/Thibaud/datasets/ultrasec/Messung8/messung8.2_labels.npy"

data_spliting="all_split"
augmentations=["addSomeNoise"]
noise_amount = 0

validation_pos = 6
data_limit = 1000
data_test_rate = 0.1    # Random % of data to run tests on (O(n**2))

num_pos = 21
num_dev = 13
signal_length = 200



############
#   Train
############
batch_size = 50
nb_epochs = 10000
patience = 100
test_interval = 500

############
#   Optim
############
optimizer = "AdamW"
sheduler = "warmup"    #"warmup"
warmup_steps = 50
learning_rate = 1e-3
lr_limit = 1e-4

###########
#   Loss
###########
lambda_distance = 14
lambda_std = 1.2
lambda_cov = 4


############
#   Model
############
model_name = "Transformer2"
latent_dimention = 200
expender_out = 128
use_extender = True
dropout = 0
# embed_size = 8 #TODO no the right numba

# Transformers
trans_embedding_size = 0 #actually becomming the multiplier of the nb of heads
trans_head_nb = 1
trans_layer_nb = 5
trans_hidden_nb = 64

# Transformer2
window_size = 16



##############
#   System
##############
device = "cuda" if torch.cuda.is_available() else "cpu"
verbose = True
plotting = False
use_wandb = True








def __get_dict__():
    __output_dic={}
    __varnames = list(globals().keys())
    for v in __varnames:
        if not v.startswith("_") and v!="torch" and v!="pprint":
            __output_dic[v] = globals()[v]

    pprint.pprint(__output_dic)
    return __output_dic


def set_parameters(args):
    for i in range(1, len(args)):
        name, value = args[i].split("=")
        name = name[2:]
        if value.isdigit():
            value = int(value)
        elif value.replace('.','',1).isdigit() and value.count('.') < 2 :
            value = float(value)
        globals()[name] = value







if __name__ == "__main__":
    __varnames = list(vars().keys())
    for v in __varnames:
        if not v.startswith("_"):
            print(v, ": \t \t ", vars()[v])