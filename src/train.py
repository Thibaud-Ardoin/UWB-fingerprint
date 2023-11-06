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

from optimizer import Optimizer, AdvOptimizer
import params
from test import testing_model
from loss import Loss

class Trainer:
    def __init__(self, trainDataloader, valDataloader, model, logger):
        self.my_model = model
        self.logger = logger
        self.trainDataloader = trainDataloader
        self.valDataloader = valDataloader
        self.loss_memory, self.cov_memory, self.dist_memory, self.var_memory, self.var_memory2, self.pos_accuracy, self.dev_accuracy = [], [], [], [], [], [], []
        self.trainLoss, self.samples = 0, 0
        self.optimizer = self.initialize_parameters()
        self.Loss = Loss(self.trainDataloader, self.my_model, self.trainLoss, self.samples, self.var_memory2, self.cov_memory, self.dist_memory, self.var_memory, self.pos_accuracy, self.dev_accuracy)
    def loss_logging(self):
        if params.loss=="crossentropy":
            self.logger.log({
            "Dev class loss": np.mean(self.var_memory),
            "Dev class accuracy": np.mean(self.dev_accuracy),
            "Encoder loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.logger.step_epoch()
        elif params.loss=="triplet+crossentropy":
            self.logger.log({
            "Triplet loss": np.mean(self.var_memory2),
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
        return optimizer
        
    def calculate_epoch_size(self):
        min_size = None
        for pos in range(len(self.trainDataloader)):
            for dev in range(len(self.trainDataloader[pos])):
                dataset_length = len(self.trainDataloader[pos][dev].dataset)
                if min_size is None or min_size > dataset_length:
                    min_size = dataset_length
        epoch_size = int(min_size / params.batch_size)
        
        return epoch_size 

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
                if params.loss=="crossentropy":
                    devLoss, self.trainLoss, self.samples, self.var_memory2, self.cov_memory, self.dist_memory, self.var_memory, self.pos_accuracy, self.dev_accuracy = self.Loss.process_flat_data()
                    self.optimizer.zero_grad()
                    devLoss.backward()
                    self.optimizer.step()
                elif params.loss=="adversarial":
                    devLoss, posLoss, encLoss, self.trainLoss, self.samples, self.var_memory2, self.cov_memory, self.dist_memory, self.var_memory, self.pos_accuracy, self.dev_accuracy = self.Loss.process_flat_data()
                    self.optimizer.zero_grad()
                    devLoss.backward(retain_graph=True)
                    posLoss.backward(retain_graph=True)
                    encLoss.backward()
                    self.optimizer.step()
            else :
                # Data divided in multi class
                for i in range(epoch_size):
                    
                    loss, self.trainLoss, self.samples, self.var_memory2, self.cov_memory, self.dist_memory, self.var_memory, self.pos_accuracy, self.dev_accuracy = self.Loss.per_epoch(epoch)
                    # Backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
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