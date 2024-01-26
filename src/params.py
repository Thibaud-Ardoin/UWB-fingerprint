"""
    Modifyable parameters file.
    Shoud use a synthax that allows grid search parametrisation too
"""
import pprint
import torch


##########
#   Data
##########
datafile = "/srv/public/Thibaud/datasets/ultrasec/Messung_9/messung9.2_data.npy"
labelfile = "/srv/public/Thibaud/datasets/ultrasec/Messung_9/messung9.2_labels.npy"

testfile = "/srv/public/Thibaud/datasets/ultrasec/Messung8/messung8.2_data.npy"
testlabelfile = "/srv/public/Thibaud/datasets/ultrasec/Messung8/messung8.2_labels.npy"

# TODO reunite properly following input types in the dataloader
data_type = "not_complex"
input_type = "fft"

data_use_position = False       # If you want to add the angular information as input of the model too

data_spliting = "pos_split"  #"all_split", "file_test", "random"
split_train_ratio = 0.80
augmentations = ["addSomeNoise"] #fourrier, logDistortionNorm
noise_amount = 0

data_limit = -1
validation_pos = [5]
validation_dev = 0      # Not used yet ?
data_test_rate = 0.01    # Random % of data to run tests on (O(n**2))

num_pos = 48    #21
num_dev = 9    #13
signal_length = 200
additional_samples = 2  # For concatenation of additional data point
same_positions = True   # If the concatenation should be done diagonal to positions or not


############
#   Train
############
batch_size = 256
nb_epochs = 10000
test_interval = 5


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
loss = "crossentropy"  #"vicreg" #"adversarial" #"triplet3" #"triplet" #"vicreg"
lambda_triplet = 10
triplet_mmargin = 1
lambda_distance = 11    #14
lambda_std = 1.2         #1.2
lambda_cov = 4          #4


############
#   Model
############
model_name = "Transformer3" #"advCNN1" #"Transformer3"
latent_dimention = 32
expender_out = 32
use_extender = True
dropout_value = 0
# embed_size = 8 #TODO no the right numba

# CNN
conv_layers_nb = 4
conv_features1_nb = 80
conv_kernel1_size = 15
conv_features2_nb = 30
conv_kernel2_size = 5
stride_size = 1
padding_size = 1
tail_fc_layers_nb = 2
feature_norm = "layer" #layer #none

expender_layers_nb = 1
expender_hidden_size = 256
class_layers_nb = 1
class_hidden_size = 256

# Transformers
trans_embedding_size = 64 #actually becomming the multiplier of the nb of heads
trans_head_nb = 2
trans_layer_nb = 3
trans_hidden_nb= 32

# Transformer2
window_size = 16



##############
#   System   #
##############
save_model = True   # On test section
use_gpu = True
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
verbose = True
plotting = False
use_wandb = False
saving_path = "./data/"



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
if loss == "AdversarialLoss" or loss == "crossentropy" or data_spliting=="random":
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

    implied_values()


def implied_values():
    # Implications on the params values
    if input_type=="fft":
        globals()["signal_length"] = globals()["signal_length"]//2
        
    # Input of model is a concatenation of signal lengthes
    globals()["signal_length"] = globals()["signal_length"] * (globals()["additional_samples"]+1)





if __name__ == "__main__":
    __varnames = list(vars().keys())
    for v in __varnames:
        if not v.startswith("_"):
            print(v, ": \t \t ", vars()[v])
