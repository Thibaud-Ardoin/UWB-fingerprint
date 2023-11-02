import sys

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

    logger = Logger()

    dg = DataGatherer()
    trainDataloader, valDataloader = dg.spliting_data()

    model = load_model()
    logger.log_model(model)

    trainer=Trainer(trainDataloader, valDataloader, model, logger)
    trainer.train()



if __name__ == "__main__":
    main()
    # wandb.agent("mwzpy9gp", main, count=5)