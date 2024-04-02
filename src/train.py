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
import time

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
        for dev in range(len(self.trainDataloader)):
            for pos in range(len(self.trainDataloader[dev])):
                dataset_length = len(self.trainDataloader[dev][pos].dataset)
                # For now make sure the dataset is full
                assert dataset_length>0, " For now this training process is accepting only a full dataset. The current label is empty: dev" + str(dev) + ", Pod" + str(pos)

                if (min_size is None or min_size > dataset_length) :
                    min_size = dataset_length
        # Min_size of dataset for each (dev, pos) pair. Each forward pass the model sees 2Pos and all devices.
        # To see about once all data, needs: (min_size x nb_pos/2)/Dataset's bsz 
        epoch_size = int(min_size*(len(self.trainDataloader[0])/2)/params.batch_size)
        
        return epoch_size 

    def train(self):

        if params.flat_data :
            epoch_size = len(self.trainDataloader)
        else :
            epoch_size = self.calculate_epoch_size()
        
        # Provide more control over granularity
        print(" * According to the training data and batch size, we need ", epoch_size, " in order to go through the whole dataset.")
        print(" * Hereby the number of steps in 1 epoch is fixed to: ", params.steps_per_epoch)
        
        # loop through the epochs
        for epoch in range(0, params.nb_epochs):
            # initialize tracker variables and set our model to trainable
            print("[INFO] epoch: {}...".format(epoch + 1))
            start_epoch_time = time.time()

            self.my_model.train()
            # Reset memorry about the last epoch loop
            self.Loss.epoch_start()

            # loop over the whole TrainLoader about 1 time
            for i in range(params.steps_per_epoch):
                if i%10==0:
                    print("Batch: ", i, "of", params.steps_per_epoch, end='\r')
                # Compile loss 
                # TODO: Most of them are crunshable -> same action = same code
                self.Loss.forwardpass_data()
                if params.loss=="CrossentropyLoss":
                    self.optimizer.zero_grad()
                    self.Loss.memory["devLoss"].backward()
                    self.optimizer.step()
                elif params.loss=="VicregLoss":
                    self.optimizer.zero_grad()
                    self.Loss.memory["loss"].backward()
                    self.optimizer.step()
                elif params.loss=="TripletLoss":
                    self.optimizer.zero_grad()
                    self.Loss.memory["loss"].backward()
                    self.optimizer.step()
                elif params.loss=="CrossTripletLoss":
                    self.optimizer.zero_grad()
                    self.Loss.memory["loss"].backward()
                    self.optimizer.step()

                elif params.loss=="AdversarialLoss":
                    # Compile Loss
                    # self.Loss.process_flat_data()
                    # Adversarial Backproploss
                    self.optimizer.zero_grad()
                    self.Loss.memory["devLoss"].backward(retain_graph=True)
                    self.Loss.memory["posLoss"].backward(retain_graph=True)
                    self.Loss.memory["encLoss"].backward()
                    self.optimizer.step()

            # Log all loss information as needed
            self.logger.log_loss(self.Loss, self.optimizer)

            epoch_time = time.time() - start_epoch_time
            trainTemplate = "TRAIN - epoch: {} train loss: {:.6f} learning rate: {:.6f} time: {:.1f}"
            print(trainTemplate.format(epoch + 1, (self.Loss.trainLoss / self.Loss.samples),
                (self.optimizer.get_lr()),
                epoch_time ))

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