import sys

import wandb

import params
from train import training_model
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

    fghdfgh

    training_model(trainDataloader, valDataloader, model, logger)




if __name__ == "__main__":
    main()
    # wandb.agent("mwzpy9gp", main, count=5)