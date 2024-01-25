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
from loss import Loss, load_loss

class Trainer:
    def __init__(self, trainDataloader, valDataloader, model, logger):
        self.my_model = model
        self.logger = logger
        self.trainDataloader = trainDataloader
        self.valDataloader = valDataloader
        self.optimizer = self.initialize_parameters()
        self.Loss = load_loss(self.trainDataloader, self.my_model)
    
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

        if params.flat_data :
            epoch_size = len(self.trainDataloader)//params.batch_size
        else :
            epoch_size = self.calculate_epoch_size()

        print(" Nb of passes per epoch: ", epoch_size)
        
        # loop through the epochs
        for epoch in range(0, params.nb_epochs):
            # initialize tracker variables and set our model to trainable
            print("[INFO] epoch: {}...".format(epoch + 1))

            self.my_model.train()
            self.Loss.epoch_start()
            # TODO: reunite flat data and multiclass

            # loop over the current batch of data
            # Flat_data is when the loss doesnt need a dataloader[dev][pos] multiclass
            if params.flat_data:    
                for i in range(epoch_size):
                    # Compile loss
                    self.Loss.forwardpass_data()
                    if params.loss=="CrossentropyLoss":
                        # Backprop
                        self.optimizer.zero_grad()
                        self.Loss.memory["devLoss"].backward()
                        self.optimizer.step()
                    elif params.loss=="VicregAdditionalSamples":
                        self.optimizer.zero_grad()
                        self.Loss.memory["loss"].backward()
                        self.optimizer.step()
                    elif params.loss=="AdversarialLoss":
                        # Compile Loss
                        self.Loss.process_flat_data()
                        # Adversarial Backproploss
                        self.optimizer.zero_grad()
                        self.Loss.memory["devLoss"].backward(retain_graph=True)
                        self.Loss.memory["posLoss"].backward(retain_graph=True)
                        self.Loss.memory["encLoss"].backward()
                        self.optimizer.step()

            else :
                # Data divided in multi class
                for i in range(epoch_size):

                    # Compile loss
                    self.Loss.per_epoch(epoch)

                    # Backprop
                    self.optimizer.zero_grad()
                    self.Loss.trainingLoss.backward()
                    self.optimizer.step()

            # Log all loss information as needed
            self.logger.log_loss(self.Loss, self.optimizer)


            trainTemplate = "TRAIN - epoch: {} train loss: {:.6f} learning rate: {:.6f}"
            print(trainTemplate.format(epoch + 1, (self.Loss.trainLoss / self.Loss.samples),
                (self.optimizer.get_lr())))

            # Optimizer
            self.optimizer.epoch_routine(self.Loss.trainLoss / self.Loss.samples)
            if self.optimizer.early_stopping() :
                testing_model(self.trainDataloader, self.valDataloader, self.my_model, self.logger)
                break
            
            # From time to time let's see wehat that models output on validation data
            if (epoch+1)%params.test_interval==0:
                testing_model(self.trainDataloader, self.valDataloader, self.my_model, self.logger)


        #end



if __name__ == "__main__":
    # test training process with dumy data
    Trainer.train()