import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import params
import torch.nn as nn

from scipy.spatial import distance_matrix
from utils.util import off_diagonal, accuracy


def normdata(x):
    x = torch.abs(x)
    x = (x - x.min())/(x.max() - x.min())
    return x

def concatenate_samples(samples, additional_samples, labels=None):
    # Concatenate every params.additional_samples samples
    number_used_sample = (additional_samples+1)*(len(samples)//(additional_samples+1))
    x = [torch.cat(tuple(samples[i:i+additional_samples+1]), dim=0) for i in range(0, number_used_sample-1, additional_samples+1)]
    if params.input_type == "fft":
        x = [normdata(torch.fft.rfft(tensor))[:-1] for tensor in x]
    
    x = torch.stack(x)
    if labels is not None:
        y = [labels[i] for i in range(0, number_used_sample-1, additional_samples+1)]
        y = torch.stack(y)
        return x, y
    return x

    
    
   
class Loss():
    def __init__(self, trainDataloader, my_model):
        self.trainDataloader = trainDataloader
        self.my_model = my_model

        self.trainLoss = 0
        self.samples = 0

        # Memorry should be sub_class specific
        self.memory_elements = ["samples", "trainLoss", "var_memory2", "cov_memory", "dist_memory", "var_memory", "dev_accuracy", "pos_accuracy", "trainAcc", "valLoss", "valAcc"]

        self.pos_amt = self.initialize_parameters()


    def epoch_start(self):
        self.build_memory_dictionary()
        self.trainLoss = 0
        self.samples = 0


    def build_memory_dictionary(self):
        self.memory = {
            elmt_name: [0] for elmt_name in self.memory_elements
        }

    def initialize_parameters(self):
        if not params.flat_data:
            pos_amt = len(self.trainDataloader[0])
        else :
            pos_amt = None
        return pos_amt
    

    def forwardpass_data(self):
        pass
    
    
    def process_flat_data(self):
       # Goes once through the dataloader
       for i, data in enumerate(self.trainDataloader):

            x, y = data
            ce_loss = nn.CrossEntropyLoss()

            if params.loss == "crossentropy":

                # if params.additional_samples > 0:
                #     x, y = concatenate_samples(x, params.additional_samples, y)
                    
                dev_pred = self.my_model(x)

                devLoss = ce_loss(dev_pred.double(), y[:,0])

                devAcc = accuracy(y[:,0], dev_pred)

                self.var_memory.append(devLoss.item())
                self.dev_accuracy.append(devAcc)

                size_of_batch = x.size(0)

                self.trainLoss += devLoss.item() * size_of_batch
                self.samples += size_of_batch


    def per_epoch(self, epoch):
        self.tripletLoss = nn.TripletMarginLoss(margin=params.triplet_mmargin, p=2, reduce=True, reduction="mean")

        if params.loss == "triplet+crossentropy":
            ce_loss = nn.CrossEntropyLoss()

            # Triplet with hard negative exemple mining
            p1 = np.random.choice(self.pos_amt)
            p2 = np.random.choice([p for p in range(self.pos_amt) if p!=p1])

            # Two data batches, with same device and different position
            batchX1 = [next(iter(self.trainDataloader[dev][p1])) for dev in range(len(self.trainDataloader))]
            batchX2 = [next(iter(self.trainDataloader[dev][p2])) for dev in range(len(self.trainDataloader))]

            encoded1 = [self.my_model.encode(batchX1[dev][0]) for dev in range(len(self.trainDataloader))]
            encoded2 = [self.my_model.encode(batchX2[dev][0]) for dev in range(len(self.trainDataloader))]

            cls_pred1 = [self.my_model.classify(enc) for enc in encoded1]
            cls_pred2 = [self.my_model.classify(enc) for enc in encoded2]

            x1 = [self.my_model.expand(enc) for enc in encoded1]
            x2 = [self.my_model.expand(enc) for enc in encoded2]

            cls_pred1 = torch.cat(cls_pred1)

            label_dev1 = torch.arange(start=0, end=len(batchX1)).to(params.device)
            label_dev1 = label_dev1.unsqueeze(1)
            label_dev1 = label_dev1.expand((len(batchX1), params.batch_size))
            label_dev1 = label_dev1.flatten()

            crossentropy = ce_loss(cls_pred1.double(), label_dev1)

            devAcc = accuracy(label_dev1, cls_pred1)

            self.var_memory.append(crossentropy.item())
            self.dev_accuracy.append(devAcc)

            # # Covariance
            # x = torch.cat(x1)
            # y = torch.cat(x2)
            # x = x - x.mean(dim=0)
            # y = y - y.mean(dim=0)

            # cov_x = (x.T @ x) / (params.batch_size - 1)
            # cov_y = (y.T @ y) / (params.batch_size - 1)
            # cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            #     self.my_model.embedding_size
            # ) + off_diagonal(cov_y).pow_(2).sum().div(self.my_model.embedding_size)


            # # TRIPLET LOSS
            distance_mat = distance_matrix(
                np.array([torch.flatten(xx1).cpu().detach().numpy() for xx1 in x1]),
                np.array([torch.flatten(xx2).cpu().detach().numpy() for xx2 in x2])
            )
            
            # Triplet loss
            trip_loss = 0
            for dev1 in range(params.num_dev):
                if epoch > 100 and np.random.rand() > 0.05:
                    neighbors = np.argpartition(distance_mat[dev1], 2)
                    dev2 = neighbors[1]
                    if dev2==dev1:
                        dev2 = neighbors[0]
                else :
                    dev2 = np.random.choice([d for d in range(params.num_dev) if d!=dev1])

                anchor = x1[dev1]         # P1, D1 
                positive = x2[dev1]       # P2, D1
                negative = x1[dev2]       # P1, D2

                trip_loss += self.tripletLoss(anchor, positive, negative)

            loss = (
                params.lambda_triplet * trip_loss / params.num_dev +
                params.lambda_class * crossentropy 
            )

            self.var_memory2.append(trip_loss.item()) # Actually triplet loss hey
            # self.cov_memory.append(cov_loss.item()) # Actually triplet loss hey
            self.var_memory.append(crossentropy.item())
            self.dev_accuracy.append(devAcc)


            size_of_batch = params.batch_size

            self.trainLoss += loss.item() * size_of_batch
            self.samples += size_of_batch

        elif params.loss == "triplet3":
            # Triplet with hard negative exemple mining and covariance regularizer
            p1 = np.random.choice(self.pos_amt)
            p2 = np.random.choice([p for p in range(self.pos_amt) if p!=p1])

            # Two data batches, with same device and different position
            batchX1 = [next(iter(self.trainDataloader[dev][p1]))[0] for dev in range(len(self.trainDataloader))]
            batchX2 = [next(iter(self.trainDataloader[dev][p2]))[0] for dev in range(len(self.trainDataloader))]

            x1 = [self.my_model(batchX1[dev]) for dev in range(len(self.trainDataloader))]
            x2 = [self.my_model(batchX2[dev]) for dev in range(len(self.trainDataloader))]

            # Covariance
            x = torch.cat(x1)
            y = torch.cat(x2)                                
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            cov_x = (x.T @ x) / (params.batch_size - 1)
            cov_y = (y.T @ y) / (params.batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                self.my_model.embedding_size
            ) + off_diagonal(cov_y).pow_(2).sum().div(self.my_model.embedding_size)


            distance_mat = distance_matrix(
                np.array([torch.flatten(xx1).cpu().detach().numpy() for xx1 in x1]),
                np.array([torch.flatten(xx2).cpu().detach().numpy() for xx2 in x2])
            )
            
            # Triplet loss
            trip_loss = 0
            for dev1 in range(params.num_dev):
                if epoch > 100 and np.random.rand() > 0.05:
                    neighbors = np.argpartition(distance_mat[dev1], 2)
                    dev2 = neighbors[1]
                    if dev2==dev1:
                        dev2 = neighbors[0]
                else :
                    dev2 = np.random.choice([d for d in range(params.num_dev) if d!=dev1])

                anchor = x1[dev1]         # P1, D1 
                positive = x2[dev1]       # P2, D1
                negative = x1[dev2]       # P1, D2

                trip_loss += self.tripletLoss(anchor, positive, negative)

            loss = (
                params.lambda_triplet * trip_loss / params.num_dev +
                params.lambda_cov * cov_loss
            )

            self.var_memory2.append(trip_loss.item()) # Actually triplet loss hey
            self.cov_memory.append(cov_loss.item()) # Actually triplet loss hey

            size_of_batch = anchor.size(0)

            self.trainLoss += loss.item() * size_of_batch
            self.samples += size_of_batch

        ####### SETUP WITH TWO RANDOM POSITIONS EVERY TIME

        elif params.loss == "triplet2":
            # Just the triplet loss but compiled on all devices 
            p1 = np.random.choice(self.pos_amt)
            p2 = np.random.choice([p for p in range(self.pos_amt) if p!=p1])
            
            # Two data batches, with same device and different position
            batchX1 = [next(iter(self.trainDataloader[dev][p1]))[0] for dev in range(len(self.trainDataloader))]
            batchX2 = [next(iter(self.trainDataloader[dev][p2]))[0] for dev in range(len(self.trainDataloader))]

            x1 = [self.my_model(batchX1[dev]) for dev in range(len(self.trainDataloader))]
            x2 = [self.my_model(batchX2[dev]) for dev in range(len(self.trainDataloader))]
            
            # Triplet loss
            trip_loss = 0
            for dev1 in range(params.num_dev):
                dev2 = np.random.choice([d for d in range(params.num_dev) if d!=dev1])

                anchor = x1[dev1]         # P1, D1 
                positive = x2[dev1]       # P2, D1
                negative = x1[dev2]       # P1, D2

                trip_loss += tripletLoss(anchor, positive, negative)

            loss = (
                params.lambda_triplet * trip_loss
            )



            self.var_memory2.append(trip_loss.item()) # Actually triplet loss hey

            size_of_batch = anchor.size(0)

            self.trainLoss += loss.item() * size_of_batch
            self.samples += size_of_batch

        ####### SETUP WITH TWO RANDOM POSITIONS EVERY TIME

        elif params.loss == "triplet":
            # Triplet loss + VICreg loss
            p1 = np.random.choice(self.pos_amt)
            p2 = np.random.choice([p for p in range(self.pos_amt) if p!=p1])
            
            # Two data batches, with same device and different position
            batchX1 = [next(iter(self.trainDataloader[dev][p1]))[0] for dev in range(len(self.trainDataloader))]
            batchX2 = [next(iter(self.trainDataloader[dev][p2]))[0] for dev in range(len(self.trainDataloader))]
            """ # Two data batches, with same device and different position  
            batchX1 = []
            batchX2 = []
            for dev in range(len(self.trainDataloader)):
                if params.validation_dev != dev:
                    p1 = np.random.choice(self.pos_amt)
                    p2 = np.random.choice([p for p in range(self.pos_amt) if p!=p1])
                else :
                    p1 = np.random.choice([p for p in range(self.pos_amt) if p!=params.validation_pos])
                    p2 = np.random.choice([p for p in range(self.pos_amt) if p!=params.validation_pos and p!=p1])
                batch1 = next(iter(self.trainDataloader[dev][p1]))[0]
                batch2 = next(iter(self.trainDataloader[dev][p2]))[0]
                batchX1.append(batch1)
                batchX2.append(batch2) """

            x1 = [self.my_model(batchX1[dev]) for dev in range(len(self.trainDataloader))]
            x2 = [self.my_model(batchX2[dev]) for dev in range(len(self.trainDataloader))]
            
#             repr_loss = F.mse_loss(x, y)
            repr_loss = torch.stack([F.mse_loss(x1[dev], x2[dev]) for dev in range(params.num_dev)]).mean()

            # Covariance
            x = torch.cat(x1)
            y = torch.cat(x2)                                
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            cov_x = (x.T @ x) / (params.batch_size - 1)
            cov_y = (y.T @ y) / (params.batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                self.my_model.embedding_size
            ) + off_diagonal(cov_y).pow_(2).sum().div(self.my_model.embedding_size)

            # Variance
            xt1 = torch.stack([x1[dev] for dev in range(len(self.trainDataloader))],  dim=0)
            xt2 = torch.stack([x2[dev] for dev in range(len(self.trainDataloader))],  dim=0)
            std_x = torch.sqrt(xt1.var(dim=0) + 0.0001)
            std_y = torch.sqrt(xt2.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            
            # Triplet loss
            dev1 = np.random.choice(params.num_dev)
            dev2 = np.random.choice([d for d in range(params.num_dev) if d!=dev1])

            anchor = x1[dev1]         # P1, D1 
            positive = x2[dev1]       # P2, D1
            negative = x1[dev2]       # P1, D2



            trip_loss = tripletLoss(anchor, positive, negative)

            loss = (
                params.lambda_triplet * trip_loss +
                params.lambda_distance * repr_loss + 
                params.lambda_cov * cov_loss + 
                params.lambda_std * std_loss
            )

            self.dist_memory.append(repr_loss.item())
            self.cov_memory.append(cov_loss.item())
            self.var_memory.append(std_loss.item()) # Actually triplet loss hey
            self.var_memory2.append(trip_loss.item()) # Actually triplet loss hey

            size_of_batch = anchor.size(0)

            self.trainLoss += loss.item() * size_of_batch
            self.samples += size_of_batch

                    ####### SETUP WITH TWO RANDOM POSITIONS EVERY TIME
                    
        elif params.loss == "vicreg":
            p1 = np.random.choice(self.pos_amt)
            p2 = np.random.choice([p for p in range(self.pos_amt) if p!=p1])
            
                # Two data batches, with same device and different position
            batchX1 = [next(iter(self.trainDataloader[dev][p1]))[0] for dev in range(len(self.trainDataloader))]
            batchX2 = [next(iter(self.trainDataloader[dev][p2]))[0] for dev in range(len(self.trainDataloader))]

            c1 = torch.concatenate(batchX1)
            c2 = torch.concatenate(batchX2)
            z1 = self.my_model(c1)
            z2 = self.my_model(c2)

            x1 = [z1[dev*params.batch_size:(dev+1)*params.batch_size] for dev in range(len(self.trainDataloader))]
            x2 = [z2[dev*params.batch_size:(dev+1)*params.batch_size] for dev in range(len(self.trainDataloader))]

            # Same as z1, z2 ?
            x = torch.cat(x1)
            y = torch.cat(x2)
            
#             repr_loss = F.mse_loss(x, y)
            repr_loss = torch.stack([F.mse_loss(x1[dev], x2[dev]) for dev in range(params.num_dev)]).sum()
            
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            size_of_batch = x.size(0)

            
#             xt1 = torch.cat([x1[dev][None] for dev in range(num_dev)],  dim=0)
#             xt2 = torch.cat([x2[dev][None] for dev in range(num_dev)],  dim=0)
            xt1 = torch.stack([x1[dev] for dev in range(len(self.trainDataloader))],  dim=0)
            xt2 = torch.stack([x2[dev] for dev in range(len(self.trainDataloader))],  dim=0)
            
#             xt1 = xt1 - xt1.mean(dim=0)
#             xt2 = xt2 - xt2.mean(dim=0)
            
            std_x = torch.sqrt(xt1.var(dim=0) + 0.0001)
            std_y = torch.sqrt(xt2.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
                
                
            # Maximise the variance allong the batch
            std_x2 = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y2 = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss2 = torch.mean(F.relu(1 - std_x2)) / 2 + torch.mean(F.relu(1 - std_y2)) / 2
            
            
            # Minimise the covariance along the encodded embeddings
            cov_x = (x.T @ x) / (params.batch_size - 1)
            cov_y = (y.T @ y) / (params.batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                self.my_model.embedding_size
            ) + off_diagonal(cov_y).pow_(2).sum().div(self.my_model.embedding_size)
            # cov_loss = F.relu(1 - cov_loss)

            # Paper's parameters 25, 25, 1
            loss = (
                params.lambda_distance * repr_loss
                + params.lambda_std * std_loss
                + params.lambda_cov * cov_loss
#                 + 25 * std_loss2
            ) / (params.lambda_distance + params.lambda_std + params.lambda_cov)

            # Keep track of the elements of the loss
            self.dist_memory.append(repr_loss.item())
            self.var_memory.append(std_loss.item())
            self.var_memory2.append(std_loss2.item())
            self.cov_memory.append(cov_loss.item())

            self.trainLoss += loss.item() * size_of_batch
            self.samples += size_of_batch
        
        return loss, self.trainLoss, self.samples, self.var_memory2, self.cov_memory, self.dist_memory, self.var_memory, self.pos_accuracy, self.dev_accuracy
    




class AdversarialLoss(Loss):
    def __init__(self, trainDataloader, my_model) -> None:
        super().__init__(trainDataloader, my_model)
        self.memory_elements.extend(["devLoss, posLoss", "encLoss", "dev_loss_memory", "pos_loss_memory", "dev_accuracy", "pos_accuracy", ])
        self.build_memory_dictionary()

        self.ce_loss = nn.CrossEntropyLoss()


    def forwardpass_data(self):
        # Extract next data batch from the dataloader
        data = next(iter(self.trainDataloader))
        x, y = data

        dev_pred, pos_pred = self.my_model(x)

        devLoss = self.ce_loss(dev_pred.double(), y[:,0])
        posLoss = self.ce_loss(pos_pred.double(), y[:,1])

        encLoss = (
            devLoss - posLoss
        )

        devAcc = accuracy(y[:,0], dev_pred)
        posAcc = accuracy(y[:,1], pos_pred)

        # Rename element of memorry correctly
        self.memory["devLoss"] = devLoss
        self.memory["posLoss"] = posLoss
        self.memory["encLoss"] = encLoss
        self.memory["dev_loss_memory"].append(devLoss.item())
        self.memory["pos_loss_memory"].append(posLoss.item())
        self.memory["pos_accuracy"].append(posAcc)
        self.memory["dev_accuracy"].append(devAcc)

        size_of_batch = x.size(0)

        self.trainLoss += encLoss.item() * size_of_batch
        self.samples += size_of_batch



class CrossentropyLoss(Loss):
    def __init__(self, trainDataloader, my_model) -> None:
        super().__init__(trainDataloader, my_model)
        self.memory_elements.extend(["devLoss", "dev_loss_memory", "dev_accuracy"])
        self.build_memory_dictionary()

        self.ce_loss = nn.CrossEntropyLoss()


    def forwardpass_data(self):
       # Goes once through the dataloader
        data = next(iter(self.trainDataloader))
        x, y = data
        # if params.additional_samples > 0:
        #     x, y = concatenate_samples(x, params.additional_samples, y)
            
        dev_pred = self.my_model(x)
        devLoss = self.ce_loss(dev_pred.double(), y[:,0])

        devAcc = accuracy(y[:,0], dev_pred)

        size_of_batch = x.size(0)
        self.trainLoss += devLoss.item() * size_of_batch
        self.samples += size_of_batch

        self.memory["devLoss"] = devLoss
        self.memory["dev_loss_memory"].append(devLoss.item())
        self.memory["dev_accuracy"].append(devAcc)




class VicregAdditionalSamples(Loss):
    def __init__(self, trainDataloader, my_model) -> None:
        super().__init__(trainDataloader, my_model)
        self.memory_elements.extend(["loss", "repr_loss_memory", "std_loss_memory", "std_loss2_memory", "cov_loss_memory"])
        self.build_memory_dictionary()


    def forwardpass_data(self):
       # Goes once through the dataloader
        data = next(iter(self.trainDataloader))

        x, y = data
        # set the number of concatenated samples for each device
        count_concatenations = params.batch_size//((params.additional_samples+1)*params.num_dev*2)

        # sfgsfg

        # x, y = concatenate_samples(x, params.additional_samples, y)
        if not params.same_positions:
            batchx1 = [element for index, element in enumerate(x) if index % 2 == 0]
            batchx2 = [element for index, element in enumerate(x) if index % 2 == 1]
            y = [element for index, element in enumerate(y) if index % 2 == 0]
            batchx1 = torch.stack(batchx1)
            batchx2 = torch.stack(batchx2)
        else :
            batchx1 = x[:len(x)//2]
            batchx2 = x[len(x)//2:]

        z1 = self.my_model(batchx1)
        z2 = self.my_model(batchx2)

        print(z1.shape)
        print(z2.shape)

        x1 = [z1[dev*(count_concatenations):(dev+1)*(count_concatenations)] for dev in range(params.num_dev)]
        x2 = [z2[dev*(count_concatenations):(dev+1)*(count_concatenations)] for dev in range(params.num_dev)]

        # Same as z1, z2 ?
        x = torch.cat(x1)
        y = torch.cat(x2)

        print(x.shape)
        print(y.shape)
        
        repr_loss = torch.stack([F.mse_loss(x1[dev], x2[dev]) for dev in range(params.num_dev)]).sum()
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        size_of_batch = x.size(0)

        xt1 = torch.stack([x1[dev] for dev in range(params.num_dev)],  dim=0)
        xt2 = torch.stack([x2[dev] for dev in range(params.num_dev)],  dim=0)
        
        std_x = torch.sqrt(xt1.var(dim=0) + 0.0001)
        std_y = torch.sqrt(xt2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
            
            
        # Maximise the variance allong the batch
        std_x2 = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y2 = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss2 = torch.mean(F.relu(1 - std_x2)) / 2 + torch.mean(F.relu(1 - std_y2)) / 2
        
        
        # Minimise the covariance along the encodded embeddings
        cov_x = (x.T @ x) / ((count_concatenations) - 1)
        cov_y = (y.T @ y) / ((count_concatenations) - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.my_model.embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.my_model.embedding_size)

        # Paper's parameters 25, 25, 1
        vic_loss = (
            params.lambda_distance * repr_loss
            + params.lambda_std * std_loss
            + params.lambda_cov * cov_loss
        ) / (params.lambda_distance + params.lambda_std + params.lambda_cov)

        self.trainLoss += vic_loss.item() * size_of_batch
        self.samples += size_of_batch


        print(repr_loss)
        print(std_loss)
        print(cov_loss)
        print(vic_loss.item())
        print(self.trainLoss)
        print(self.samples)
        
        self.memory["vicLoss"] = vic_loss
        self.memory["repr_loss_memory"].append(repr_loss.item())
        self.memory["std_loss_memory"].append(std_loss.item())
        self.memory["std_loss2_memory"].append(std_loss2.item())
        self.memory["cov_loss_memory"].append(cov_loss.item())

def load_loss(trainLoader, Model):
	loss_object = eval(params.loss)(trainLoader, Model)
	if params.verbose:
		print(loss_object)
	return loss_object

