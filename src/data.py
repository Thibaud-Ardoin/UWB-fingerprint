"""
    data.py
    Load the data, preprocessing, dataloader definition
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools

import params

import torch
import torchvision.transforms as transforms
import torchaudio


from custom_batchsampler import CustomBatchSampler

############################
#   Processing functions
############################
def center_data_gradient(d):
    target_position = 40

    # Compute the derivative of the absolute value of the signal
    derivative = np.gradient(np.abs(d)) 

    # Find the index where the derivative is highest
    max_derivative_idx = np.argmax(derivative) 

    # Calculate the needed shift
    shift_amount = target_position - max_derivative_idx

    # Shift the sample
    shifted_sample = np.roll(d, shift_amount)

    return shifted_sample

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

def normalize_tensor(x):
    min_val = x.min()
    max_val = x.max()
    range_val = max_val - min_val
    x = (x - min_val) / range_val
    return x

def fourier(x):
    x = torch.fft.fft(x)
    return x

def spectrogram(x):
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=params.spectrogram_window_size, hop_length=8, power=1, normalized=True, onesided=False).to(params.device)
    x = spectrogram(x)
    return x

def rfft(x):
    return normdata(torch.fft.rfft(x))[:-1]


def add_angular(data_point, label_point):
    # INPUT:    data_point: 250p signal  X label_point: (device id, position id) 
    angle = label_point[1] * 2 * np.pi / params.num_pos
    
    # OUTPUT:   data_point: (250p signal, 250*cos(a), 250*sin(a))
    cos = torch.full_like(data_point[:, 0], torch.cos(angle))
    sin = torch.full_like(data_point[:, 0], torch.sin(angle))
    
    return torch.stack((data_point[:, 0], data_point[:, 1], cos, sin))

def logDistortionNorm(x):
    eps = 1e-5    
    logX1 = np.log(x+eps)
    # return logX1
    return (logX1 - logX1.min())/(logX1.max() - logX1.min())

# Function to apply Hann window
def apply_hann_window(x):
    hann_window = torch.hann_window(len(x))
    return x * hann_window

def random_shift(x, clean_test=False):
    # Fill with 50 random time_steps additionally randomly around the current
    filling_size = 80
    x = torch.view_as_real(x)
    x = x - torch.mean(x)
    if clean_test or np.random.rand() < 0.1:
        # Alligned and zeros
        a = filling_size//2
        xx = torch.zeros((filling_size + len(x), 2))
        xx[a:(len(x) + a)] = x
    else :
        # Random shift with random padding
        xx = torch.rand((filling_size + len(x), 2)) * torch.mean(torch.abs(x))
        a = int(np.random.rand()*filling_size)
        xx[a:(len(x) + a)] = x

    return torch.view_as_complex(xx)


#################################################
#   Dataset definition for the training process
#################################################


# Own dataset type, specially to take my data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, testset=False):
        self.testset = testset
        self.data = data
        self.transform_list = [
            lambda x: center_data_gradient(x),
            lambda x: torch.from_numpy(x)    
        ]
        self.augmentations = [
            eval(function_name) for function_name in params.augmentations
        ] 

        # If not a testset, we add the desired augmentations
        if not self.testset:
            self.transform_list += self.augmentations
        elif self.testset and "random_shift" in params.augmentations:
            self.transform_list += [lambda x: random_shift(x, clean_test=True)]

        # 0 padding for the ViT model
        if params.model_name == "ViT":
            if params.input_type != "spectrogram":
                self.transform_list += [lambda x: torch.cat((x, torch.zeros(6, dtype=x.dtype)), dim=0)]
            else:
                if params.additional_samples > 0:
                    self.transform_list += [lambda x: torch.cat((x, torch.zeros(5, dtype=x.dtype)), dim=0)]

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
        if params.data_use_position:
            x = add_angular(x, y)

        return x, y
    
    def __len__(self):
        return len(self.data)
    


########################################################################################################
#   DataLoader to draw data in good batches according ot the number to desired mesurments per datapoint.
########################################################################################################


class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data_set, batch_size=params.batch_size, additional_samples=params.additional_samples, same_positions=params.same_positions):
        self.nb_concatenated = additional_samples + 1

        balanced_batch_sampler = CustomBatchSampler(data_set, additional_samples=additional_samples, same_positions=same_positions, batch_size=batch_size*self.nb_concatenated)
        super(MyDataLoader, self).__init__(dataset=data_set, batch_sampler=balanced_batch_sampler)

        self.post_concat_transform_list = []
        if params.input_type == "spectrogram":
            self.post_concat_transform_list += [lambda x: spectrogram(x)]
        elif params.input_type == "fft":
            self.post_concat_transform_list += [lambda x: fourier(x)]
        if params.data_type == "complex":
            self.post_concat_transform_list += [torch.view_as_real]
        else:
            self.post_concat_transform_list += [lambda x: torch.abs(x)]

        self.post_concat_transform_list += [lambda x: normalize_tensor(x), lambda x: x.to(torch.float32)]

        self.post_concat_transforms = transforms.Compose(
            self.post_concat_transform_list
        )

    def concatenate_samples(self, samples, labels=None):
        # Concatenate every params.additional_samples samples
        number_used_sample = (self.nb_concatenated)*(len(samples)//(self.nb_concatenated))
        x = [torch.cat(torch.unbind(samples[i:i+self.nb_concatenated]), dim=0) for i in range(0, number_used_sample, self.nb_concatenated)]
        x = torch.stack(x).to(params.device)

        if labels is not None:
            # ! Select only 1st label of each group of concatenated signals
            y = [labels[i] for i in range(0, number_used_sample, self.nb_concatenated)]
            y = torch.stack(y).to(params.device)
            return x, y
        return x


    def __iter__(self):
        for x, y in super(MyDataLoader, self).__iter__():
            x, y = self.concatenate_samples(x, y)
            x = self.post_concat_transforms(x)

            yield x, y
        # folded_labels = y.reshape(y.shape[0]//self.nb_concatenated, self.nb_concatenated, 2)



###########################################################
#   DataGatherer collect from source and format the data
###########################################################

class DataGatherer():
    def __init__(self):

        # Loading the data from source file
        self.data = np.load(params.datafile)
        self.labels = np.load(params.labelfile)
        # self.data_formating()

        if params.data_spliting == "file_test":
            self.test_data = np.load(params.testfile)
            self.test_label = np.load(params.testlabelfile)


    def data_formating(self):
        # Formating of Label data
        # initial_position_number = 15
        # initial_device_number = 3

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
        
        elif params.data_spliting == "file_test" or params.data_spliting == "file_split":
            return self.file_split()
        else :
            return self.positional_split(return_array)
        
    def file_split(self):
        """
            Split the data according to a test file and a training file.
            In our case, we train with our latest quality data and see if it generalise with our previous measurmments:
            This means Different time of recording and environment
        """ 
        training_loader = self.positional_split(its_all_train=True)

        # Gather all the test data as a linear data loader

        sub_set_indices = np.random.choice(len(self.test_data), int(params.data_test_rate*len(self.test_data)), replace=False)
        z = list(zip(self.test_data[sub_set_indices], self.test_label[sub_set_indices]))

        test_set = MyDataset(z, testset=True)
        test_loader = MyDataLoader(test_set)

        return training_loader, test_loader




    def random_split(self):
        """
            Just randomly separate the all data into test and train
            enable to see performances in the case of learned position
            Also available in non flat case: with a random split of each dev/pos pair
        """
        z = list(zip(self.data, self.labels))
        if params.flat_data:
            # Random split the whole data into test and train
            ind = np.arange(len(z))
            np.random.shuffle(ind)
            if params.data_limit > 0:
                ind = np.random.choice(ind, min(len(ind), params.data_limit), replace=False)
            data_size = len(ind)

            train_data = [z[ind[i]] for i in range(int(params.split_train_ratio*data_size))]
            val_data = [z[ind[i]] for i in range(int(params.split_train_ratio*data_size), data_size)]

            training_set = MyDataset(train_data, testset=False)
            training_loader = MyDataLoader(training_set)

            validation_set = MyDataset(val_data, testset=True)
            validation_loader = MyDataLoader(validation_set)

        else:   # Return loaders that matrixes of loaders
            # split each dev/pair subset into test/train
            pos_ids = [np.where(self.labels[:,1] == i) for i in range(params.num_pos)]
            dev_ids = [np.where(self.labels[:,0] == i) for i in range(params.num_dev)]
            
            training_loader = []
            validation_loader = []
            # For each device, put the first part of the data in training, the second part in validation
            for i in range(params.num_dev) :
                training_loader.append([])
                validation_loader.append([])
                for k in range(params.num_pos) :
                    # Make the union between dev and pos befor getting the time separation
                    inter_ids = list(set(dev_ids[i][0]) & set(pos_ids[k][0]))
                    if params.data_limit > 0:
                        inter_ids = np.random.choice(inter_ids, min(len(inter_ids), params.data_limit), replace=False)

                    # split(i, k) indices in train testsplit in

                    [z[inter_ids[j]] for j in range(len(inter_ids))]


                    training_loader.append()
                    validation_loader.append()

                    training_set = MyDataset()
                    training_loader[dev].append(MyDataLoader(training_set))


        return training_loader, validation_loader



    def positional_split(self, return_array=False, its_all_train=False):
        """
            Separating the data such as the train and test data doesnt incorporate the same positions.
            This will enable us to test the generalisation of our model.
        """

        # Gather the indexes for the diferent poses and the different devices
        z = list(zip(self.data, self.labels))
        if params.verbose:
            print(" \t * Current total loaded raw data reach size:", len(z))
            print(" \t   Note that some might be removed according to the used Devices for example.")
        pos_ids = [np.where(self.labels[:,1] == i) for i in range(params.num_pos)]
        dev_ids = [np.where(self.labels[:,0] == i) for i in range(params.num_dev)]

        # Hard coded test to have a proper visual of the used labels
        if False: 
            m = []
            for dev in range(params.num_dev):
                m.append([])
                for pos in range(params.num_pos):
                    ids = list(set(dev_ids[dev][0]) & set(pos_ids[pos][0]))
                    m[dev].append(len(ids))

            m = np.array(m)
            plt.imshow(m)
            plt.colorbar()
            plt.show()

        all_data = []
        if its_all_train:
            # Remove all validation position, this is a pure training set
            params.validation_pos = []

        # For each device, put the first part of the data in training, the second part in validation
        for i in range(params.num_dev) :
            all_data.append([])
            for k in range(params.num_pos) :
                # Make the union between dev and pos befor getting the time separation
                inter_ids = list(set(dev_ids[i][0]) & set(pos_ids[k][0]))
                if params.data_limit > 0:
                    inter_ids = np.random.choice(inter_ids, min(len(inter_ids), params.data_limit), replace=False)
                all_data[i].append([z[inter_ids[j]] for j in range(len(inter_ids))])

        # Gather the training elements
        training_loaders = []
        for dev in range(params.num_dev) :
            training_loaders.append([])
            for pos in range(params.num_pos) :
                if not pos in params.validation_pos :
                    if params.flat_data:
                        # Dont divide in multi data loader for each class combination
                        training_loaders = training_loaders + all_data[dev][pos]
                    else :
                        training_set = MyDataset(all_data[dev][pos])
                        training_loaders[dev].append(MyDataLoader(training_set))
        if params.flat_data:
            # Sanity check
            for i in range(len(training_loaders)-1, -1, -1):
                if len(training_loaders[i]) == 0:
                    del training_loaders[i]
            training_loaders = MyDataLoader(MyDataset(training_loaders))

        if its_all_train:
            return training_loaders

        # Gather the unique validation position
        val_data = []
        for dev in range(params.num_dev) :
            for vali_pos in params.validation_pos:
                val_data = val_data + all_data[dev][vali_pos]
        validation_set = MyDataset(val_data, testset=True)
        # Adjust batch size according to the feedforward type
        bsz = params.batch_size
        if params.loss=="VicregLoss": bsz = bsz * params.num_dev
        validation_loader = MyDataLoader(validation_set, batch_size=bsz)

        self.training_loaders = training_loaders
        self.validation_loader = validation_loader
        if return_array:
            return training_loaders, validation_loader, all_data
        return training_loaders, validation_loader
    


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