"""
    Modifyable parameters file.
    Shoud use a synthax that allows grid search parametrisation too
"""
import torch


##########
#   Data
##########
datafile = "/srv/public/Thibaud/datasets/ultrasec/Messung8/messung8.2_data.npy"
labelfile = "/srv/public/Thibaud/datasets/ultrasec/Messung8/messung8.2_labels.npy"

data_spliting="all_split"
augmentations=[]

validation_pos = 2
data_limit = 1000

num_pos = 21
num_dev = 13



############
#   Train
############
batch_size = 64
nb_epochs = 500
patience = 100
test_interval = 50
learning_rate = 1e-3

lambda_distance = 25
lambda_std = 25
lambda_cov = 5


############
#   Model
############
model_name = "EncoderNet"
latent_dimention = 32
expender_out = 64
# embed_size = 8 #TODO no the right numba
trans_embedding_size = 32
trans_head_nb = 4
trans_layer_nb = 4
trans_hidden_nb = 64



##############
#   System
##############
device = "cuda" if torch.cuda.is_available() else "cpu"
verbose = False
plotting = False








def __get_dict__():
    __output_dic={}
    __varnames = list(globals().keys())
    for v in __varnames:
        if not v.startswith("_") and v!="torch":
            __output_dic[v] = globals()[v]

    print(__output_dic)
    return __output_dic







if __name__ == "__main__":
    __varnames = list(vars().keys())
    for v in __varnames:
        if not v.startswith("_"):
            print(v, ": \t \t ", vars()[v])