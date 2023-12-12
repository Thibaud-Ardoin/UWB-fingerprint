"""
    data.py
    Load the data, preprocessing, dataloader definition
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
import torchvision.transforms as transforms

import params
import torchaudio
import librosa

############################
#   Processing functions
############################

def addSomeNoise(x):
    if params.noise_amount > 0:
        # Add some random noise to the data, 1/x of amount max
        x = x + torch.rand(x.shape)*params.noise_amount
    return x

def zeroPadding(x):
    # select a reandom span of the signal that is padded down to 0
    r = np.random.rand(3)
    if r[0]>0.9 :
        max_length = 25
        length = int(max_length*r[1])
        start = int((200-length)*r[2])
        x[start:start+length] = 0
    return x

def normdata(x):
    x = torch.abs(x)
    x = (x - x.min())/(x.max() - x.min())
    return x

def fourier(x):
    x = torch.fft.fft(x)
    x = torch.abs(x)
    return x


#################################################
#   Dataset definition for the training process
#################################################


# Own dataset type, specially to take my data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, testset=False):
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=100, hop_length=12)

        self.testset = testset
        self.data = data

        if params.input_type == "fft":
            self.transform_list = [
                lambda x: np.fft.fft(x),
                lambda x: normdata(x),
                lambda x: torch.from_numpy(x),
                lambda x: x.to(torch.float32)
            ]
        elif params.input_type == "spectrogram":
            self.transform_list = [
                lambda x: torch.from_numpy(x),
                lambda x: spectrogram(x),
                lambda x: x.to(torch.float32)
            ]
        elif params.input_type == "raw":
            self.transform_list = [
                lambda x: torch.from_numpy(x),
                lambda x: x.to(torch.float32)
            ]
        elif params.input_type == "fft_complex":
            self.transform_list = [
                lambda x: torch.from_numpy(x),
                lambda x: torch.fft.rfft(x),
                lambda x: x.to(torch.complex64)
            ]

        self.augmentations = [
            eval(function_name) for function_name in params.augmentations
        ] 

        # If not a testset, we add the desired augmentations
        if not self.testset:
            self.transform_list += self.augmentations

        self.transforms = transforms.Compose(
            self.transform_list
        )
        
    def __getitem__(self, index):
        try:
            x, y = self.data[index]
        except Exception as e:
            print(e)
            print(len(self.data), index)
            print(self.data[index])
            x, y = self.data[index-1]
        x = self.transforms(x).to(params.device)
        y = torch.from_numpy(y).to(params.device)

        return x, y
    
    def __len__(self):
        return len(self.data)


###########################################################
#   DataGatherer collect from source and format the data
###########################################################

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
class DataGatherer():
    def __init__(self):

        # Loading the data from source file
        self.data = np.load(params.datafile)
        self.labels = np.load(params.labelfile)
        # self.data_formating()


    def data_formating(self):
        # Formating of Label data
        initial_position_number = 15
        initial_device_number = 3

        formated_data = list(self.data)
        formated_labels = list(self.labels)
        if params.verbose:
            print("Initial dataset length:", len(formated_labels), len(formated_data))

        # Removing the corrupted parts of data and label
        failed_positions = [9, 12]
        fail_pos_ids = [np.where(self.labels[:,1] == i)[0] for i in failed_positions]
        fail_pos_ids = list(itertools.chain(*fail_pos_ids))
        if len(fail_pos_ids) > 0:
            srtd_poses = sorted(fail_pos_ids, reverse=True)
            for ind in srtd_poses:
                formated_data.pop(ind)
                formated_labels.pop(ind)
        if params.verbose:
            print("New dataset length:", len(formated_labels), len(formated_data))

        # Attribution of new labels to get a clean suit of label from 0 to new_position_number -1
        new_position_number = initial_position_number - len(failed_positions)
        # Create a simple projection to the new labels
        new_label_proj = []
        ind = 0
        for i in range(initial_position_number) :
            new_label_proj.append(ind)
            if not i in failed_positions :
                ind += 1

        # Projecting position labels
        formated_labels = np.array(list(map(lambda lab: [lab[0], new_label_proj[lab[1]]], formated_labels)))
        formated_data = np.array(formated_data)

        self.data = formated_data
        self.labels = formated_labels

        if params.plotting:
            plt.title(" Labels: the devices and the positions")
            plt.plot(self.labels)
            plt.show()

        if params.verbose:
            print("This process removed the undesired position labels", failed_positions)
            print("The new position labels are all the integers from 0 to", max(new_label_proj), " for position")

        # num_pos = new_position_number
        # num_dev = 3


    def spliting_data(self, return_array=False):
        if params.data_spliting == "random":
            return self.random_split()
        else :
            return self.positional_split(return_array)

    def random_split(self):
        """
            Just randomly separate the all data into test and train
            enable to see performances in the case of learned position
        """
        z = list(zip(self.data, self.labels))
        ind = np.arange(len(z))
        np.random.shuffle(ind)
        if params.data_limit > 0:
            ind = np.random.choice(ind, min(len(ind), params.data_limit))
        data_size = len(ind)

        train_data = [z[ind[i]] for i in range(int(params.split_train_ratio*data_size))]
        val_data = [z[ind[i]] for i in range(int(params.split_train_ratio*data_size), data_size)]

        training_set = MyDataset(train_data, testset=False)
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=params.batch_size, shuffle=True)

        validation_set = MyDataset(val_data, testset=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=params.batch_size, shuffle=True)
        
        return training_loader, validation_loader



    def positional_split(self, return_array=False):
        """
            Separating the data such as the train and test data doesnt incorporate the same positions.
            This will enable us to test the generalisation of our model.
        """

        # Gather the indexes for the diferent poses and the different devices
        z = list(zip(self.data, self.labels))
        pos_ids = [np.where(self.labels[:,1] == i) for i in range(params.num_pos)]
        dev_ids = [np.where(self.labels[:,0] == i) for i in range(params.num_dev)]
        all_data = []

        # For each device, put the first part of the data in training, the second part in validation
        for i in range(params.num_dev) :
            all_data.append([])
            for k in range(params.num_pos) :
                # Make the union between dev and pos befor getting the time separation
                inter_ids = list(set(dev_ids[i][0]) & set(pos_ids[k][0]))
                if params.data_limit > 0:
                    inter_ids = np.random.choice(inter_ids, min(len(inter_ids), params.data_limit))
                all_data[i].append([z[inter_ids[j]] for j in range(len(inter_ids))])

        # Gather the training elements
        training_loaders = []
        for dev in range(params.num_dev) :
            training_loaders.append([]) # for flat data we add an empty list?
            for pos in range(params.num_pos) :
                if not pos==params.validation_pos :
                    if params.flat_data:
                        # TODO: make it cleaner
                        # list of potential positions of the additional measurements
                        choices = [x for x in range(params.num_pos) if x != params.validation_pos]
                        for i in range(len(all_data[dev][pos])):
                            positions = np.array([])
                            if params.multiple_train_positions:
                                # randomly select the additional positions
                                positions = np.random.choice(choices, params.additional_samples, replace=False)
                            else:
                                positions = np.tile([pos], params.additional_samples)
                            concatenated_data = np.array(all_data[dev][pos][i][0])
                            if not positions.size == 0: 
                                for y in positions:
                                    # randomly select the indexes of the additional measurements
                                    random = list(np.random.choice(len(all_data[dev][y]), 1, replace=False))
                                    # concatenate the samples
                                    concatenated_data = np.append(concatenated_data, all_data[dev][y][random[0]][0])
                            training_loaders.append([concatenated_data, all_data[dev][pos][i][1]])

                        # old way
                        # Dont divide in multi data loader for each class combination
                        #training_loaders = training_loaders + all_data[dev][pos]

                    else :
                        training_set_helper = []
                        for i in range(len(all_data[dev][pos])):  
                            concatenated_data = np.array(all_data[dev][pos][i][0])
                            random = np.array([])
                            random = np.random.choice(len(all_data[dev][pos]), params.additional_samples, replace=False)
                            if not random.size == 0:
                                for y in random:
                                    concatenated_data = np.append(concatenated_data, all_data[dev][pos][y][0])
                            training_set_helper.append([concatenated_data, all_data[dev][pos][i][1]]) 
                        training_set = MyDataset(training_set_helper)

                        # old way
                        #training_set = MyDataset(all_data[dev][pos])
                        training_loaders[dev].append(torch.utils.data.DataLoader(training_set, batch_size=params.batch_size, shuffle=True))
        if params.flat_data:
            # Sanity check
            for i in range(len(training_loaders)-1, -1, -1):
                if len(training_loaders[i]) == 0:
                    del training_loaders[i]
            training_loaders = torch.utils.data.DataLoader(MyDataset(training_loaders), batch_size=params.batch_size, shuffle=True)


        # Gather the unique validation position
        # TODO: Make it also a multi positional element
        val_data = []
        for dev in range(params.num_dev) :
            #val_data = val_data + all_data[dev][params.validation_pos]
            for i in range(len(all_data[dev][params.validation_pos])):
                concatenated_data = np.array(all_data[dev][params.validation_pos][i][0])
                if params.multiple_test_measurements:
                    mes = np.array([])
                    # randomly select the indexes of the additional measurements
                    mes = np.random.choice(len(all_data[dev][params.validation_pos]), params.additional_samples, replace=False)
                    if not mes.size == 0:
                        for y in mes:
                            # concatenate the samples
                            concatenated_data = np.append(concatenated_data, all_data[dev][params.validation_pos][y][0])
                else:
                    if params.additional_samples > 0:
                        for _ in range(params.additional_samples):
                            concatenated_data = np.append(concatenated_data, all_data[dev][params.validation_pos][i][0])
                val_data.append([concatenated_data, all_data[dev][params.validation_pos][i][1]])
        validation_set = MyDataset(val_data, testset=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=params.batch_size, shuffle=True)
        self.training_loaders = training_loaders
        self.validation_loader = validation_loader
        if return_array:
            return training_loaders, validation_loader, all_data
        return training_loaders, validation_loader
        

    # def data_loading():
    #     data, label = load_form_source()


if __name__ == "__main__":

    # Testing the dataloading with visualisation ect
    dg = DataGatherer()
    training_loaders, validation_loader, data_array = dg.spliting_data(return_array=True)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=100, hop_length=12)
    print("** Dataset characterisation **", "\n")

    print("Number of devices:", len(data_array))
    for dev in range(params.num_dev):
        print("Number of positions:", len(data_array[dev]))
        for pos in range(params.num_pos):
            print("Size of data of specific dev and pos:", len(data_array[dev][pos]))
            print("Dim of each data element (absolute complex amplitude):", len(data_array[dev][pos][0][0]))
    #         print("Dim of each complex element:", len(data_array[dev][pos][0][0][0]))
            print("Dim of each label (dev, pos):", len(data_array[dev][pos][0][1]))
            break
        break

    # dev group, pos group, (elmt id), (data, labels), (dim1, dim 2) X (dev label, pos label)
    print("Print label for device N°2 and position N°3", data_array[2][3][0][1])

    print("Mean impulse response shape")
    mean_pulse = np.array([[[data_array[0][0][i][0] for i in range(1000)] for dev in range(params.num_dev)] for pos in range(params.num_pos)])
    print(mean_pulse.shape)
    mean_pulse = mean_pulse.reshape(mean_pulse.shape[0]*mean_pulse.shape[1]*mean_pulse.shape[2], 200)
    mean_pulse = mean_pulse.mean(axis=0)
    # print(np.array(mean_pulse).tolist())
    plt.plot(mean_pulse)
    plt.show()