"""
    Modifyable parameters file.
    Shoud use a synthax that allows grid search parametrisation too
"""
import pprint
import torch


##########
#   Data
##########
datafile = "/data/Messung_9/messung9.raw.3_data.npy"
labelfile = "/data/Messung_9/messung9.raw.3_labels.npy"

testfile = "/data/Messung_10/messung10.raw.2_data.npy"
testlabelfile = "/data/Messung_10/messung10.raw.2_labels.npy"

data_type = "not_complex"
input_type = "spectrogram" #rfft"
spectrogram_type = "fourier" #fourier
spectrogram_window_size=32
spectrogram_hop_size = 8
data_use_position = False       # If you want to add the angular information as input of the model too

data_spliting = "pos_split"  #"all_split", "file_test", "random"
split_train_ratio = 0.80
augmentations = ["addSomeNoise"] #fourrier, logDistortionNorm
noise_amount = 0
shift_added_size = 50

data_limit = -1
validation_pos = [4, 5]
validation_dev = []     
data_test_rate = 0.05    # Random ratio of data to run tests on (O(n**2))

num_pos = 48    #21
num_dev = 13    #13
signal_length = 250
additional_samples = 0  # For concatenation of additional data point
same_positions = True   # If the concatenation should be done diagonal to positions or not


############
#   Train
############
batch_size = 32
steps_per_epoch = 100   # Provide data independant granularity of the training process
nb_epochs = 1000
test_interval = 25


############
#   Optim
############
optimizer = "Adam"
sheduler = "combi"    #"warmup" plateau "combi" for the combination of both
warmup_steps = 50
learning_rate = 1e-3
lr_limit = 1e-4
patience = 50


###########
#   Loss
###########
loss = "CrossentropyLoss"  #"VicregLoss" #"AdversarialLoss" #"CrossentropyLoss" #"TripletLoss"
triplet_mmargin = 1
lambda_triplet = 1
lambda_class = 1
lambda_distance = 11    #14
lambda_std = 4         #1.2
lambda_cov = 4          #4


############
#   Model
############
model_name = "ClassCNN1" #"Transformer3" #"advCNN1" #"Transformer3" "ConvMixer" "ClassCNN1" "ViT"
latent_dimention = 256
expender_out = 256
use_extender = True
dropout_value = 0

# Arccos/Arcface
arcface = True
arcface_margin = 0.1
arcface_scale = 16

# Expender
expender_layers_nb = 1
expender_hidden_size = 256

# Classification
class_layers_nb = 1
class_hidden_size = 256

# CNN
conv_layers_nb = 4
conv_features1_nb = 80
conv_kernel1_size = 15
conv_features2_nb = 30
conv_kernel2_size = 5
stride_size = 1
cnn_padding_size = 1
tail_fc_layers_nb = 2
feature_norm = "layer" #layer #none
pooling_kernel_size = 4
pooling_stride_size = 2


# Transformers
trans_embedding_size = 10 #actually becomming the multiplier of the nb of heads
trans_head_nb = 5
trans_layer_nb = 4
trans_hidden_nb= 128
trans_padding_size = 0

# ConvMixer
convm_embedding_size = 128     # 128
convm_layer_nb = 10           # 4
convm_kernel_size = 25        # 25
convm_patch_size = 10         # 15
convm_out_size = 32           # 64

# Transformer2
window_size = 16



##############
#   System   #
##############
num_workers = 1
save_model = True   # On test section
use_gpu = True
device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
verbose = True
plotting = False
use_wandb = True
saving_path = "./data/"
saved_model_suffix = "some_testing"


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
if loss == "AdversarialLoss" or loss == "CrossentropyLoss" or data_spliting=="random":
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
    if input_type=="rfft":
        globals()["signal_length"] = globals()["signal_length"]//2

    if loss=="VicregLoss":
        globals()["batch_size"] = globals()["batch_size"]//globals()["num_dev"]

    if "random_shift" in globals()["augmentations"] or "random_insertions" in globals()["augmentations"] or "random_shift_insert" in globals()["augmentations"]:
        globals()["signal_length"] = globals()["signal_length"] +  globals()["shift_added_size"]

    # Input of model is a concatenation of signal lengthes
    globals()["signal_length"] = globals()["signal_length"] * (globals()["additional_samples"]+1)


def save_to_yaml():
    pass




if __name__ == "__main__":
    __varnames = list(vars().keys())
    for v in __varnames:
        if not v.startswith("_"):
            print(v, ": \t \t ", vars()[v])
