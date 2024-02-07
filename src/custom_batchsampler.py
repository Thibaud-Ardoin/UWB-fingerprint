import numpy as np
import torch
import time

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import params


def custom_sort_key(item):
    values = item.split()
    return int(values[0]), int(values[1])


class CustomBatchSampler(BatchSampler):
    
    """
    CustomBatchSampler is a specialized BatchSampler that generates batches with a specific configuration, 
    where the batch size is divided by the number of additional samples to ensure a diverse set of classes in each batch. 
    It supports obtaining x samples in sequence with the same label or the same label and the same position.
    
    Args:
    - dataset (Dataset): The dataset to sample from.
    - additional_samples (int): Number of additional samples per class in each batch.
    - same_positions (bool): Flag indicating whether to consider the position similarity of labels.
    - batch_size (int): Total size of the batch.
    - check (bool, optional): Flag to enable batch checking. Defaults to False.
    """

    # def __init__(self, dataset, additional_samples, same_positions, batch_size, check=False):
    def __init__(self, dataset, additional_samples, same_positions, batch_size, 
                 loss=params.loss, 
                 validation_pos=params.validation_pos, 
                 num_dev=params.num_dev, 
                 num_pos=params.num_pos, 
                 train=False, 
                 check=False):

    #     loader = DataLoader(dataset)
        self.labels = []
        self.count = 0
        self.count_samples = additional_samples+1
        self.dataset = dataset
        self.batch_size = batch_size
        # set the number of classes in each batch
        self.count_classes = batch_size//(additional_samples+1)
        self.same_positions = same_positions
        self.loss = loss
        self.train = train
        self.validation_pos = validation_pos
        self.num_dev = num_dev
        self.num_pos = num_pos
        # check the batches for mistakes
        self.check = check
        self.list_indicies = []
        #  for same posititions we need this multiplier to get the right number of samples (bachsize) for vicreg 
        if self.loss == 'VicregAdditionalSamples' and self.train and self.same_positions:
            self.count_multiplier = self.batch_size//(self.count_samples*self.num_dev*2)
        # for different posititions we need to double the number because we can easily divide the batch 
        elif self.loss == 'VicregAdditionalSamples' and self.train and not self.same_positions:
            self.count_multiplier = self.batch_size//(self.count_samples*self.num_dev)
        else:
            self.count_multiplier = 1
        # Process labels based on position similarity
        if same_positions:
            # Convert list of label pairs as a tensor
            for _, label in self.dataset.data:
                self.labels.append(label)
            self.unique_labels = set(map(tuple, self.labels))
            self.labels = torch.tensor(np.array(self.labels))

            self.unique_labels = [' '.join(map(str, label)) for label in self.unique_labels]
            self.indicies_for_label = {
                label: np.where(
                    (self.labels.cpu().numpy()[:,0] == np.fromstring(label, sep=' ').astype(int)[0]) & 
                    (self.labels.cpu().numpy()[:,1] == np.fromstring(label, sep=' ').astype(int)[1])
                )[0]
                for label in self.unique_labels
            }
        else:
            # Keeping only device information from the labels
            for _, label in self.dataset.data:
                self.labels.append(label[0])
            self.labels = torch.tensor(np.array(self.labels))
            self.unique_labels = list(set(self.labels.numpy()))
            self.indicies_for_label = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.unique_labels}
        if self.check:
            print(self.indicies_for_label)
            print(len(self.indicies_for_label))

        # Shuffle indices for each unique label
        for l in self.unique_labels:
            np.random.shuffle(self.indicies_for_label[l])
        
        # Initialize counters for used indices for each unique label
        self.indicies_used_labels = {label: 0 for label in self.unique_labels}


    def __iter__(self):
        self.count = 0
        self.list_indicies = []
        while self.count + self.batch_size < len(self.dataset):
            # test for VicregAdditionalSamples training
            if self.loss == 'VicregAdditionalSamples' and self.train and self.same_positions:
                excluded_elements = self.validation_pos
                remaining_elements = list(set(range(self.num_pos)) - set(excluded_elements))
                # Randomly choose a label in range of position used for training 
                pos1 = np.random.choice(remaining_elements)
                remaining_elements.remove(pos1)
                pos2 = np.random.choice(remaining_elements)
                # Sort the list based on the custom sorting key

                # WHY sorting ? isnt it just a np.where(label[:,1]==pos1) situation ?
                # AH, sorting in order to have the right device labels at each place ??
                # But then, what if there is a decalage in the number of devices for same position ??
                # 

                sorted_list = sorted(self.unique_labels, key=custom_sort_key)
                result_list1 = [item for item in sorted_list if item.split()[1] == str(pos1)]
                result_list2 = [item for item in sorted_list if item.split()[1] == str(pos2)]
                selected_classes = result_list1 + result_list2
            elif self.loss == 'VicregAdditionalSamples' and self.train and not self.same_positions:
                # Randomly select count_classes labels for the current batch
                selected_classes = self.unique_labels
            else:
                # Randomly select count_classes labels for the current batch
                selected_classes = np.random.choice(self.unique_labels, self.count_classes)

            # Initialize an empty list to store indices for the current batch
            batch_indices = []

            # Iterate over the selected classes
            for current_class in selected_classes:
                # Get indices for the current class
                current_class_indices = self.indicies_for_label[current_class]

                # Extract a subset of indices for the current class
                subset_indices = current_class_indices[
                    self.indicies_used_labels[current_class]:
                    self.indicies_used_labels[current_class] + self.count_samples *self.count_multiplier
                ]

                # Extend the batch indices with the subset
                batch_indices.extend(subset_indices)

                # Update the count of used indices for the current class
                self.indicies_used_labels[current_class] += self.count_samples*self.count_multiplier

                # Check if all indices for the current class have been used
                if (
                    self.indicies_used_labels[current_class] + self.count_samples*self.count_multiplier
                    > len(current_class_indices)
                ):
                    # If so, shuffle the indices for the current class
                    np.random.shuffle(current_class_indices)

                    # Reset the count of used indices for the current class
                    self.indicies_used_labels[current_class] = 0

            # Append the batch indices to the list of all batch indices
            self.list_indicies.append(batch_indices)

            # Yield the indices for the current batch
            yield batch_indices

            # Update the overall count based on the number of classes and samples
            self.count += self.count_classes * self.count_samples*self.count_multiplier
        # check the batches for mistakes
        if self.check:
            for i, x in enumerate(self.list_indicies):
                print(i,x,len(x))
            print(len(self.dataset))


    def __len__(self):
        return len(self.dataset) // self.batch_size
    


if __name__ == "__main__":

    ###########################
    # LETs TRY THIS SAMPLER
    ##############################
    
    from data import MyDataset, MyDataLoader
    import params
    import matplotlib.pyplot as plt
    from models import ClassCNN1
    import time

    number_data_points = 1500
    data_size = 200
    bsz = 128
    data = np.random.rand(number_data_points, data_size)
    nb_dev = 3
    nb_pos = 5
    params.batch_size = bsz 
    params.num_dev = nb_dev
    params.num_pos = nb_pos
    labels = [[[(dev, pos) for _ in range(100)] for dev in range(nb_dev)] for pos in range(nb_pos) ]
    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0]*labels.shape[1]*labels.shape[2], labels.shape[3]))

    print(labels.shape)
    print(data.shape)

    z = list(zip(data, labels))
    dataset = MyDataset(z)

    if False:   # Test the classic dataloader, basically random shuffle of all data
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)
    if False:   # Test the customBatchSampler with normal loader. Need concatenation
        sampler = CustomBatchSampler(dataset, additional_samples=2, same_positions=True, batch_size=bsz, check=False)
        dataLoader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    if True:   # Test customBatchWith MyDataLoader
        params.signal_length = params.signal_length*(1+params.additional_samples)
        dataLoader = MyDataLoader(dataset, additional_samples=2, same_positions=False, batch_size=bsz)

    print("wait.....")

    model = ClassCNN1().to(params.device)

    for d in dataLoader:
        x, y = d
        print(x.shape)
        print(y.shape)
        # plt.plot(x[0].cpu().numpy())
        # plt.show()
        out = model(x)
        print(out.shape)
        # plt.plot(out.detach().cpu().numpy())
        # plt.show()

        break