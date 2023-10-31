"""
    Training script leading to the diferent optimisation, loss, epoch handler

    Currently out of hand, but needs to be reorganised. The outline here beeing only 
        - Load training tools
        - loop on loader
        - apply loss function
            - Loss linnks to encapsulated classes for the different types
        - Log it
"""
import numpy as np

from scipy.spatial import distance_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from optimizer import Optimizer, AdvOptimizer
import params
from test import testing_model
from utils.util import accuracy, off_diagonal


class Trainer:
    def __init__(self, trainDataloader, valDataloader, model, logger):
        self.my_model = model
        self.logger = logger
        self.trainDataloader = trainDataloader
        self.valDataloader = valDataloader
        self.loss_memory, self.cov_memory, self.dist_memory, self.var_memory, self.var_memory2, self.pos_accuracy, self.dev_accuracy = [], [], [], [], [], [], []
        self.trainLoss, self.trainAcc ,self.valLoss, self.valAcc, self.samples = 0, 0, 0, 0, 0
        self.optimizer, self.tripletLoss, self.ce_loss = self.initialize_parameters()
    def loss_logging(self):
        if params.loss=="crossentropy":
            self.logger.log({
            "Dev class loss": np.mean(self.var_memory),
            "Dev class accuracy": np.mean(self.dev_accuracy),
            "Encoder loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.logger.step_epoch()
        elif params.loss=="adversarial":
            self.logger.log({
            "Dev class loss": np.mean(self.var_memory),
            "Pos class loss": np.mean(self.cov_memory),
            "Pos class accuracy": np.mean(self.pos_accuracy), 
            "Dev class accuracy": np.mean(self.dev_accuracy),
            "Encoder loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.logger.step_epoch()
        elif params.loss=="triplet3":
            self.logger.log({
            "triploss": np.mean(self.var_memory2),
            "cov_loss": np.mean(self.cov_memory),
            "global_loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.logger.step_epoch()
        elif params.loss=="triplet2":
            self.logger.log({
            "triploss": np.mean(self.var_memory2),
            "global_loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.logger.step_epoch()
        elif params.loss=="triplet":
            self.logger.log({
            "repr_loss": np.mean(self.dist_memory),
            "cov_loss": np.mean(self.cov_memory),
            "std_loss": np.mean(self.var_memory),
            "triploss": np.mean(self.var_memory2),
            "global_loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.logger.step_epoch()
        elif params.loss=="vicreg":
            self.logger.log({"repr_loss": np.mean(self.dist_memory),
            "std_loss": np.mean(self.var_memory),
            "std_loss2": np.mean(self.var_memory2),
            "cov_loss": np.mean(self.cov_memory),
            "global_loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.logger.step_epoch()
    def initialize_parameters(self):
        if params.model_name == "advCNN1":
            optimizer = AdvOptimizer(self.my_model)
        else:
            optimizer = Optimizer(self.my_model)

        if params.loss.startswith("triplet"):
            triplet_loss = nn.TripletMarginLoss(margin=params.triplet_mmargin, p=2, reduce=True, reduction="mean")
            ce_loss = None
        elif params.loss in ["adversarial", "crossentropy"]:
            ce_loss = nn.CrossEntropyLoss()
            triplet_loss = None

        return optimizer, triplet_loss, ce_loss
    def loss_function(self, epoch):
        if params.loss == "triplet3":
            # Triplet with hard negative exemple mining
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

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

                trip_loss += self.tripletLoss(anchor, positive, negative)

            loss = (
                params.lambda_triplet * trip_loss
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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



            trip_loss = self.tripletLoss(anchor, positive, negative)

            loss = (
                params.lambda_triplet * trip_loss +
                params.lambda_distance * repr_loss + 
                params.lambda_cov * cov_loss + 
                params.lambda_std * std_loss
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.trainLoss += loss.item() * size_of_batch
            self.samples += size_of_batch      
    def calculate_epoch_size(self):
        min_size = None
        for pos in range(len(self.trainDataloader)):
            for dev in range(len(self.trainDataloader[pos])):
                dataset_length = len(self.trainDataloader[pos][dev].dataset)
                if min_size is None or min_size > dataset_length:
                    min_size = dataset_length
        epoch_size = int(min_size / self.params.batch_size)
        
        return epoch_size 
    def process_flat_data(self):
       for i, data in enumerate(self.trainDataloader):

            x, y = data

            if params.loss == "adversarial":

                dev_pred, pos_pred = self.my_model(x)

                devLoss = self.ce_loss(dev_pred.double(), y[:,0])
                posLoss = self.ce_loss(pos_pred.double(), y[:,1])

                encLoss = (
                    devLoss - posLoss
                )

                # Backprop
                self.optimizer.zero_grad()

                devLoss.backward(retain_graph=True)
                posLoss.backward(retain_graph=True)
                encLoss.backward()

                self.optimizer.step()

                devAcc = accuracy(y[:,0], dev_pred)
                posAcc = accuracy(y[:,1], pos_pred)

                self.var_memory.append(devLoss.item())
                self.cov_memory.append(posLoss.item())
                self.pos_accuracy.append(posAcc)
                self.dev_accuracy.append(devAcc)
                # self.var_memory2.append(trip_loss.item())

                size_of_batch = x.size(0)

                self.trainLoss += encLoss.item() * size_of_batch
                self.samples += size_of_batch

            elif params.loss == "crossentropy":

                dev_pred = self.my_model(x)

                devLoss = self.ce_loss(dev_pred.double(), y[:,0])

                # Backprop
                self.optimizer.zero_grad()
                devLoss.backward()
                self.optimizer.step()

                devAcc = accuracy(y[:,0], dev_pred)

                self.var_memory.append(devLoss.item())
                self.dev_accuracy.append(devAcc)

                size_of_batch = x.size(0)

                self.trainLoss += devLoss.item() * size_of_batch
                self.samples += size_of_batch
    def train(self):
        encodded_validations = []

        if not params.flat_data :
            epoch_size = self.calculate_epoch_size()
            print(" Nb of passes per epoch: ", epoch_size)
        
        # loop through the epochs
        for epoch in range(0, params.nb_epochs):
            # initialize tracker variables and set our model to trainable
            print("[INFO] epoch: {}...".format(epoch + 1))

            self.my_model.train()
            
            # loop over the current batch of data
            if params.flat_data:
                self.process_flat_data()

            else :
                # Data divided in multi class
                for i in range(epoch_size):
                    # Chose two different positions
                    pos_amt = len(self.trainDataloader[0])
                    self.loss_function(epoch)
            
            self.loss_logging()


            trainTemplate = "TRAIN - epoch: {} train loss: {:.6f} learning rate: {:.6f}"
            print(trainTemplate.format(epoch + 1, (self.trainLoss / self.samples),
                (self.optimizer.get_lr())))
            self.loss_memory.append(self.trainLoss / self.samples)

            # Optimizer
            self.optimizer.epoch_routine(self.trainLoss / self.samples)
            if self.optimizer.early_stopping() :
                break
            
            # From time to time let's see wehat that models output on validation data
            if (epoch+1)%params.test_interval==0:
                testing_model(self.trainDataloader, self.valDataloader, self.my_model, self.logger)

        testing_model(self.trainDataloader, self.valDataloader, self.my_model, self.logger)

        #end
        return self.loss_memory, self.cov_memory, self.var_memory, self.var_memory2, self.dist_memory, encodded_validations



if __name__ == "__main__":
    # test training process with dumy data
    Trainer.train()