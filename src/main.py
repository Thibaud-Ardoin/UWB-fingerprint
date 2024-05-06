import sys
import time

import wandb

import params

"""
    This is the main execusion code of the uwb_fingerprints git.
    One might want to 

"""

def main():

    # Set right parametrisation
    params.set_parameters(sys.argv)

    # Imports
    from train import Trainer
    from test import testing_model
    from models import load_model
    from data import DataGatherer
    from logger import Logger

    # Logger
    t1 = time.time()
    logger = Logger()
    t2 = time.time()
    print("Logger creation time: ", t2-t1)

    # Data
    dg = DataGatherer()
    trainDataloader, valDataloader = dg.spliting_data()
    t3 = time.time()
    print("DataLoaders creation time: ", t3-t2)

    # Model
    model = load_model()
    logger.log_model(model)
    t4 = time.time()
    print("Model creation time: ", t4-t3)

    # Train
    trainer=Trainer(trainDataloader, valDataloader, model, logger)
    trainer.train()

    logger.finish()



if __name__ == "__main__":
    main()