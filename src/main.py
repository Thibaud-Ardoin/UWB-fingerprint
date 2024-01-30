import sys
import time

import wandb

import params
from train import Trainer
from test import testing_model
from models import load_model
from data import DataGatherer
from logger import Logger



"""
    This is the main execusion code of the uwb_fingerprints git.
    One might want to 

"""

def main():

    params.set_parameters(sys.argv)

    t1 = time.time()
    logger = Logger()
    t2 = time.time()
    print("Logger creation time: ", t2-t1)

    dg = DataGatherer()
    trainDataloader, valDataloader = dg.spliting_data()
    t3 = time.time()
    print("DataLoaders creation time: ", t3-t2)


    model = load_model()
    # logger.log_model(model)
    t4 = time.time()
    print("Model creation time: ", t4-t3)

    trainer=Trainer(trainDataloader, valDataloader, model, logger)
    trainer.train()

    logger.finish()



if __name__ == "__main__":
    main()
    # wandb.agent("mwzpy9gp", main, count=5)