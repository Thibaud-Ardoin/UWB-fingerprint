import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import params

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

    def __init__(self, dataset, additional_samples, same_positions, batch_size, check=False):
        loader = DataLoader(dataset)
        self.labels = []
        self.count = 0
        self.count_samples = additional_samples+1
        self.dataset = dataset
        self.batch_size = batch_size
        # set the number of classes in each batch
        self.count_classes = batch_size//(additional_samples+1)
        self.same_positions = same_positions
        # check the batches for mistakes
        self.check = check
        self.list_indicies = []

        # Process labels based on position similarity
        if same_positions:
            for _, label in loader:
                self.labels.append(label[0])
            self.labels = torch.stack(self.labels)
            #TODO: workaround for the fact that sets don't work with numpy arrays
            self.unique_labels = set(map(tuple, self.labels.cpu().numpy()))
            self.unique_labels = [' '.join(map(str, label)) for label in self.unique_labels]
            self.indicies_for_label = {
                label: np.where(
                    (self.labels.cpu().numpy()[:,0] == np.fromstring(label, sep=' ').astype(int)[0]) & 
                    (self.labels.cpu().numpy()[:,1] == np.fromstring(label, sep=' ').astype(int)[1])
                )[0]
                for label in self.unique_labels
}
        else:
            for _, label in loader:
                self.labels.append(label[0][0])
            self.labels = torch.LongTensor(self.labels)
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
                    self.indicies_used_labels[current_class] + self.count_samples
                ]

                # Extend the batch indices with the subset
                batch_indices.extend(subset_indices)

                # Update the count of used indices for the current class
                self.indicies_used_labels[current_class] += self.count_samples

                # Check if all indices for the current class have been used
                if (
                    self.indicies_used_labels[current_class] + self.count_samples
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
            self.count += self.count_classes * self.count_samples
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

    number_data_points = 1500
    data_size = 200
    bsz = 128
    data = np.random.rand(number_data_points, data_size)
    nb_dev = 3
    nb_pos = 5
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
        dataLoader = MyDataLoader(dataset, additional_samples=2, same_positions=False, batch_size=bsz)

    for d in dataLoader:
        x, y = d
        print(x.shape)
        print(y.shape)
        plt.plot(y.cpu().numpy())
        plt.show()
        break