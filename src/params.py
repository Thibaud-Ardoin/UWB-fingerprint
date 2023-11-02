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

data_spliting = "random"  #"all_split"
split_train_ratio = 0.80
augmentations = ["addSomeNoise"]
noise_amount = 0

data_limit = 500000
validation_pos = 5
validation_dev = 0      # Not used yet ?
data_test_rate = 0.1    # Random % of data to run tests on (O(n**2))

num_pos = 21    #21
num_dev = 13    #13
signal_length = 200



############
#   Train
############
batch_size = 1024
nb_epochs = 10000
test_interval = 1000


############
#   Optim
############
optimizer = "Adam"
sheduler = "plateau"    #"warmup" plateau
warmup_steps = 50
learning_rate = 1e-3
lr_limit = 1e-4
patience = 50


###########
#   Loss
###########
loss = "crossentropy" #"adversarial" #"triplet3" #"triplet" #"vicreg"
triplet_mmargin = 1
lambda_distance = 14    #14
lambda_std = 1.2         #1.2
lambda_cov = 1          #4
lambda_triplet = 10


############
#   Model
############
model_name = "ClassCNN1" #"advCNN1" #"Transformer3"
latent_dimention = 64
expender_out = 64
use_extender = False
dropout_value = 0
# embed_size = 8 #TODO no the right numba

# CNN
conv_layers_nb = 4
conv_features1_nb = 32
conv_kernel1_size = 5
conv_features2_nb = 64
conv_kernel2_size = 3
stride_size = 2
padding_size = 2
tail_fc_layers_nb = 2
feature_norm = "batch" #layer #none

class_layers_nb = 3
class_hidden_size = 64

# Transformers
trans_embedding_size = 32 #actually becomming the multiplier of the nb of heads
trans_head_nb = 1
trans_layer_nb = 1
trans_hidden_nb = 32

# Transformer2
window_size = 16



##############
#   System   #
##############
use_gpu = True
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
verbose = True
plotting = False
use_wandb = True



###########################
#   Include Yaml config   #
###########################
def __use_config__(file_name):
    import yaml
    with open(file_name, "r") as f:
        __config = yaml.load(f, Loader=yaml.FullLoader)
        for k in __config:
            globals()[k] = __config[k]


#########################
#   Implied values
#########################
# These are variables that are concequences of some previous combinations

flat_data = False
if loss == "adversarial" or data_spliting=="random":
    flat_data = True



def __get_dict__():
    __output_dic={}
    __varnames = list(globals().keys())
    for v in __varnames:
        if not v.startswith("_") and not v in ["torch", "yaml", "pprint"]:
            __output_dic[v] = globals()[v]

    pprint.pprint(__output_dic)
    return __output_dic


def set_parameters(args):
    for i in range(1, len(args)):
        name, value = args[i].split("=")
        name = name[2:]
        if name == "config":
            __use_config__(value)
        else:
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
