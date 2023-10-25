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

validation_pos = 0
validation_dev = 0
data_limit = 2000
data_test_rate = 0.1    # Random % of data to run tests on (O(n**2))

num_pos = 21    #21
num_dev = 13    #13
signal_length = 200



############
#   Train
############
batch_size = 128
nb_epochs = 10000
test_interval = 1

############
#   Optim
############
optimizer = "Adam"
sheduler = "warmup"    #"warmup" plateau
warmup_steps = 50
learning_rate = 1e-3
lr_limit = 1e-4
patience = 100

###########
#   Loss
###########
loss = "adversarial" #"adversarial" #"triplet3" #"triplet" #"vicreg"
triplet_mmargin = 1
lambda_distance = 14    #14
lambda_std = 1.2         #1.2
lambda_cov = 1          #4
lambda_triplet = 10


############
#   Model
############
model_name = "advCNN1" #"advCNN1" #"Transformer3"
latent_dimention = 32
expender_out = 32
use_extender = False
dropout_value = 0
# embed_size = 8 #TODO no the right numba

# CNN


# Transformers
trans_embedding_size = 32 #actually becomming the multiplier of the nb of heads
trans_head_nb = 1
trans_layer_nb = 1
trans_hidden_nb = 32

# Transformer2
window_size = 16



##############
#   System
##############
use_gpu = True
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
verbose = True
plotting = False
use_wandb = True



#########################
#   Implied values
#########################
# These are variables that are concequences of some previous combinations

flat_data = False
if loss == "adversarial":
    flat_data = True


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