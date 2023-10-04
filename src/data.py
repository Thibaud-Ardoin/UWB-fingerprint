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



############################
#   Processing functions
############################

def addSomeNoise(x):
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


#################################################
#   Dataset definition for the training process
#################################################


# Own dataset type, specially to take my data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, testset=False):
        self.testset = testset
        self.data = data
        self.transform_list = [
            lambda x: torch.from_numpy(x),
            lambda x: x.to(torch.float32)
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
        x, y = self.data[index] 
        x = self.transforms(x).to(params.device)
            
        return x, y
    
    def __len__(self):
        return len(self.data)


###########################################################
#   DataGatherer collect from source and format the data
###########################################################

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
                inter_ids = np.random.choice(list(set(dev_ids[i][0]) & set(pos_ids[k][0])), params.data_limit)
                all_data[i].append([z[inter_ids[j]] for j in range(len(inter_ids))])

        # Gather the training elements
        training_loaders = []
        for dev in range(params.num_dev) :
            training_loaders.append([])
            for pos in range(params.num_pos) :
                if not pos==params.validation_pos :
                    training_set = MyDataset(all_data[dev][pos])
                    training_loaders[dev].append(torch.utils.data.DataLoader(training_set, batch_size=params.batch_size, shuffle=True))

        # Gather the unique validation position
        # TODO: Make it also a multi positional element
        val_data = []
        for dev in range(params.num_dev) :
            val_data = val_data + all_data[dev][params.validation_pos]
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