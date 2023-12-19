"""
    logger.py
    Ment to encapsulate a w&b style of library in order to save all the logs of our training
    the mmetrics, models outputs, random seed, artefacts, weights
"""
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

import torch
import wandb
from torchviz import make_dot

import params



class Logger():
    def __init__(self):
        if params.use_wandb:
            self.setupWB()
        self.epoch=0
        self.test_step = 0

    def step_epoch(self):
        self.epoch += 1

    def step_test(self):
        self.test_step += 1


    def setupWB(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="Fingerprint1",
            
            # track hyperparameters and run metadata
            config=params.__get_dict__()
        )

    def log_model(self, model):
        if not params.model_name.startswith("adv"): 
            if params.data_use_position:
                x = torch.rand(params.batch_size, 4, params.signal_length)
            else :
                x = torch.rand(params.batch_size, params.signal_length)
            y = model(x.to(params.device))
            make_dot(y, params=dict(list(model.named_parameters()))).render("data/torchviz_test", format="jpg")

        
    def log(self, info, name=None):
        if params.use_wandb:
            if type(info)==dict:
                info["epoch"]= self.epoch
                wandb.log(info)
            else :
                if name is not None and type(name)==str:
                    wandb.log({name: info, "epoch": self.epoch})
                else:
                    wandb.log({"unkown": info, "epoch": self.epoch})


    def log_curve(self, curve, title="some curve", column_names=["x", "y"]):
        if params.use_wandb:
            table = wandb.Table(data=curve, columns=column_names)
            wandb.log({
                title + "_ep" + str(self.test_step): wandb.plot.line(table, column_names[0], column_names[1], title=title + str(self.test_step)),
                "epoch": self.epoch
            })


    def log_scatter(self, data, labels, title="Some Scatter"):
        if params.use_wandb:
            X = np.array(data)
            dimentions = [[2*i, i*2+1] for i in range(1)] #len(X[0])//2)]
            colors_base = cm.rainbow(np.linspace(0, 1, params.num_dev))
            plt.rcParams["figure.figsize"] = (20,20)
            # colors = [colors_base[l] for l in labels]
            # print(colors_base)
            # print(np.arange(0, params.num_dev+1))
            cmap, norm = matplotlib.colors.from_levels_and_colors(np.arange(0, params.num_dev+1)- 0.5, colors_base)

            # print(cmap)
            # print(norm)
            
            for i in range(len(dimentions)):
                fig, ax = plt.subplots()
                scatter = ax.scatter(X[:, dimentions[i][0]], X[:, dimentions[i][1]], c=labels, norm=norm, cmap=cmap, marker=".", linewidths=0.5, s=20)
                handles, lab = scatter.legend_elements(prop="colors", num=params.num_dev, alpha=0.6)
                legend1 = ax.legend(handles, lab,
                        loc="lower left", title="Device id")
                ax.add_artist(legend1)
                ax.set_title("dimentions " + str(i*2) + "and " + str(i*2+1))
                # .legend(loc='upper left',prop = {'size':7},bbox_to_anchor=(1,1))
                # plt.tight_layout(pad=5)
                plot_img = "data/plot_"+str(i)+".png"
                fig.savefig(plot_img)
                wandb.log({
                    title + str(i) + "_ep" + str(self.test_step): wandb.Image(plot_img),
                    "epoch": self.epoch
                })
                plt.close()

    def finish(self):
        if params.use_wandb:
            wandb.finish()



if __name__ == "__main__":
    logger = Logger()
    logger.log(12, "numba")
    logger.log(13, "numba")
    logger.log(15, "numba")
    logger.log("an other type of data ball")
    logger.finish()