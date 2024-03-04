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

import plotly
import torch
import wandb
from torchviz import make_dot

import params

from sklearn.decomposition import PCA


class Logger():
    def __init__(self):
        if params.use_wandb:
            self.setupWB()
        self.epoch=0
        self.test_step = 0
        self.max_memory = {}

    def step_epoch(self):
        self.epoch += 1

    def step_test(self):
        self.test_step += 1


    def save_model(self, model):
        if params.save_model:
            torch.save(model, params.saving_path + "model_" + params.saved_model_suffix + "_" + str(self.epoch) + ".pth")

    def log_loss(self, loss, optim):
        # TODO setup an automated logging taht logs all the loss.memory content
        if params.loss=="CrossentropyLoss":
            self.log({
            "Dev class loss": np.mean(loss.memory["dev_loss_memory"]),
            "Dev class accuracy": np.mean(loss.memory["dev_accuracy"]),
            "Encoder loss": loss.trainLoss / loss.samples,
            "learning rate": optim.get_lr()})
            self.step_epoch()
        if params.loss=="VicregLoss":
            self.log({"repr_loss": np.mean(loss.memory["repr_loss_memory"]),
            "std_loss": np.mean(loss.memory["std_loss_memory"]),
            "std_loss2": np.mean(loss.memory["std_loss2_memory"]),
            "cov_loss": np.mean(loss.memory["cov_loss_memory"]),
            "global_loss": loss.trainLoss / loss.samples,
            "learning rate": optim.get_lr()})
            self.step_epoch()
        if params.loss=="TripletLoss":
            self.log({"triplet_loss": np.mean(loss.memory["triplet_loss_memory"]),
            "cov_loss": np.mean(loss.memory["cov_loss_memory"]),
            "global_loss": loss.trainLoss / loss.samples,
            "learning rate": optim.get_lr()})
            self.step_epoch()
        if params.loss=="CrossTripletLoss":
            self.log({"triplet_loss": np.mean(loss.memory["triplet_loss_memory"]),
            "crossentropy_loss": np.mean(loss.memory["crossentropy_loss_memory"]),
            "dev_accuracy": np.mean(loss.memory["dev_accuracy"]),
            "global_loss": loss.trainLoss / loss.samples,
            "learning rate": optim.get_lr()})
            self.step_epoch()

        elif params.loss=="triplet+crossentropy":
            self.log({
            "Triplet loss": np.mean(self.var_memory2),
            "Dev class loss": np.mean(self.var_memory),
            "Dev class accuracy": np.mean(self.dev_accuracy),
            "Encoder loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.step_epoch()
        elif params.loss=="AdversarialLoss":
            self.log({
            "Dev class loss": np.mean(loss.memory["dev_loss_memory"]),
            "Pos class loss": np.mean(loss.memory["pos_loss_memory"]), 
            "Dev class accuracy": np.mean(loss.memory["dev_accuracy"]),
            "Pos class accuracy": np.mean(loss.memory["pos_accuracy"]),
            "Encoder loss": loss.trainLoss / loss.samples,
            "learning rate": optim.get_lr()})
            self.step_epoch()
        elif params.loss=="triplet3":
            self.log({
            "triploss": np.mean(self.var_memory2),
            "cov_loss": np.mean(self.cov_memory),
            "global_loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.step_epoch()
        elif params.loss=="triplet2":
            self.log({
            "triploss": np.mean(self.var_memory2),
            "global_loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.step_epoch()
        elif params.loss=="triplet":
            self.log({
            "repr_loss": np.mean(self.dist_memory),
            "cov_loss": np.mean(self.cov_memory),
            "std_loss": np.mean(self.var_memory),
            "triploss": np.mean(self.var_memory2),
            "global_loss": self.trainLoss / self.samples,
            "learning rate": self.optimizer.get_lr()})
            self.step_epoch()
        elif params.loss=="vicreg":
            self.log({"repr_loss": np.mean(loss.dist_memory),
            "std_loss": np.mean(loss.var_memory),
            "std_loss2": np.mean(loss.var_memory2),
            "cov_loss": np.mean(loss.cov_memory),
            "global_loss": loss.trainLoss / loss.samples,
            "learning rate": optim.get_lr()})
            self.step_epoch()



    def setupWB(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="Fingerprint1",
            
            # track hyperparameters and run metadata
            config=params.__get_dict__()
        )

    def log_model(self, model):
        nb_params = sum(p.numel() for p in model.parameters())
        self.log({"Number Parameters": nb_params})
        if False :
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

    def log_maximisation_value(self, info):
        for k in info:
            max_k = "max_" + k
            val = info[k]
            if max_k in self.max_memory.keys():
                if self.max_memory[max_k] < val:
                    self.max_memory[max_k] = val
            else:
                self.max_memory[max_k] = val
                    
        self.log(self.max_memory)



    def log_curve(self, curve, title="some curve", column_names=["x", "y"]):
        if params.use_wandb:
            table = wandb.Table(data=curve, columns=column_names)
            wandb.log({
                title + "_ep" + str(self.test_step): wandb.plot.line(table, column_names[0], column_names[1], title=title + str(self.test_step)),
                "epoch": self.epoch
            })


    def log_scatter(self, data, labels, title="Some Scatter"):
        if params.use_wandb:
            num_classes = max(labels)
            X = np.array(data)

            #PCA
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(data)

            colors_base = cm.rainbow(np.linspace(0, 1, num_classes+1))
            plt.rcParams["figure.figsize"] = (20,20)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=labels, cmap='rainbow', marker=".", linewidths=0.5, s=20)
            handles, lab = scatter.legend_elements(prop="colors", num=num_classes+1, alpha=0.8)
            legend1 = ax.legend(handles, lab, loc="lower left", title="Device id")
            ax.add_artist(legend1)
     
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plot_img = "data/plot_3d.png"
            plt.tight_layout()
            fig.savefig(plot_img)
            wandb.log({
                title + "_ep" + str(self.test_step): wandb.Image(plot_img), #plt
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

    logger.log_maximisation_value({"yoo": 1})
    logger.log_maximisation_value({"yoo": 2})
    logger.log_maximisation_value({"yoo": 3})
    logger.log_maximisation_value({"yoo": 1})
    logger.log_maximisation_value({"yoo": 0})
    logger.log_maximisation_value({"yoo": -1})
    logger.finish()