"""
    Training script leading to the diferent optimisation, loss, epoch handler
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau

import params
from test import testing_model

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def training_model(t_loader, v_loader, mymodel, logger):
    encodded_validations = []
    optimizer = optim.Adam(mymodel.parameters(), lr=params.learning_rate)
    tripletLoss = nn.TripletMarginLoss(p=1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params.patience)
    
    min_size = None
    for pos in range(len(t_loader)) :
        for dev in range(len(t_loader[pos])):
            if min_size is None or min_size > len(t_loader[pos][dev].dataset) :
                min_size = len(t_loader[pos][dev].dataset)
                
    epoch_size = int(min_size/params.batch_size)
    print(" Nb of passes per epoch: ", epoch_size)
    
    # loop through the epochs
    for epoch in range(0, params.nb_epochs):
        # initialize tracker variables and set our model to trainable
        print("[INFO] epoch: {}...".format(epoch + 1))
        trainLoss = 0
        trainAcc = 0
        valLoss = 0
        valAcc = 0
        samples = 0
        mymodel.train()
        loss_memory, cov_memory, dist_memory, var_memory, var_memory2 = [], [], [], [], []

        # loop over the current batch of data
        
        for i in range(epoch_size):
            # Chose two different positions
            pos_amt = len(t_loader[0])
            
            ####### SETUP WITH ALL POSITINS SEEN
#             rndm_pos = np.arange(pos_amt)
#             np.random.shuffle(rndm_pos)
                    
#             out_all = []
#             for j in range(len(rndm_pos)):
#                 batchX = []
#                 for d in range(len(t_loader)):
#                     out_p = []
#                     p = rndm_pos[j]
#                     batchX.append(next(iter(t_loader[d][p]))[0])
#                 z = mymodel(torch.cat(batchX, dim=0))
#                 zl = [z[dev*param.batch_size:(dev+1)*param.batch_size] for dev in range(len(t_loader))]
#                 out_all.append(torch.stack(zl, dim=1))

#             # Create two vectors of the same data but with onaligned positions
#             # Batch x Position x Device x Embd
#             x_pos_proj1 = torch.stack(out_all, dim=1)
#             x_pos_proj2 = torch.stack(out_all[1:] + [out_all[0]], dim=1)
            
# #             x_pos_proj1 = x_pos_proj1 - x_pos_proj1.mean(dim=0)
# #             x_pos_proj2 = x_pos_proj2 - x_pos_proj2.mean(dim=0)
            
#             repr_loss = F.mse_loss(x_pos_proj1, x_pos_proj2)#, reduction='none')
# #             repr_loss = torch.sum(repr_loss, dim=-1)
    
#             x_pos_proj1 = x_pos_proj1 - x_pos_proj1.mean(dim=0)
    
# #             xt1 = torch.cat([x_pos_proj1[:, dev][None] for dev in range(num_dev)],  dim=0)
# #             std_x = torch.sqrt(x_pos_proj1.var(dim=1) + 0.0001)
# #             std_loss = torch.mean(F.relu(1 - std_x))

#             cov_loss = []
#             std_loss = []
#             for pos in range(len(x_pos_proj1[0])):
                
#                 std_x = torch.sqrt(x_pos_proj1[:, pos].var(dim=1) + 0.0001)
#                 std_loss.append(torch.mean(F.relu(1 - std_x)))

#                 for dev in range(len(x_pos_proj1[0][0])):
#                     cov_x = (x_pos_proj1[:, pos, dev].T @ x_pos_proj1[:, pos, dev]) / (param.batch_size - 1)
#                     cov_loss.append(off_diagonal(cov_x).pow_(2).sum().div(
#                         embed_size
#                     ))
#             cov_loss = torch.stack(cov_loss).mean()
#             std_loss = torch.stack(std_loss).mean()
        
#             size_of_batch = x_pos_proj1.size(0)
#             loss = (
#                 25 * repr_loss
#                 + 25 * std_loss
#                 + 1 * cov_loss
#             )


            ####### SETUP WITH TWO RANDOM POSITIONS EVERY TIME
            
            p1 = np.random.choice(pos_amt)
            p2 = np.random.choice([p for p in range(pos_amt) if p!=p1])
            
                # Two data batches, with same device and different position
            batchX1 = [next(iter(t_loader[dev][p1]))[0] for dev in range(len(t_loader))]
            batchX2 = [next(iter(t_loader[dev][p2]))[0] for dev in range(len(t_loader))]

            c1 = torch.concatenate(batchX1)
            c2 = torch.concatenate(batchX2)
            z1 = mymodel.encoder(c1)
            z2 = mymodel.encoder(c2)

            x1 = [z1[dev*params.batch_size:(dev+1)*params.batch_size] for dev in range(params.num_dev)]
            x2 = [z2[dev*params.batch_size:(dev+1)*params.batch_size] for dev in range(params.num_dev)]

            x = torch.cat(x1)
            y = torch.cat(x2)
            
#             repr_loss = F.mse_loss(x, y)
            repr_loss = torch.stack([F.mse_loss(x1[dev], x2[dev]) for dev in range(params.num_dev)]).mean()
            
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            size_of_batch = x.size(0)

            
#             xt1 = torch.cat([x1[dev][None] for dev in range(num_dev)],  dim=0)
#             xt2 = torch.cat([x2[dev][None] for dev in range(num_dev)],  dim=0)
            xt1 = torch.stack([x1[dev] for dev in range(params.num_dev)],  dim=0)
            xt2 = torch.stack([x2[dev] for dev in range(params.num_dev)],  dim=0)
            
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
                params.trans_embedding_size
            ) + off_diagonal(cov_y).pow_(2).sum().div(params.trans_embedding_size)

            # Paper's parameters 25, 25, 1
            loss = (
                params.lambda_distance * repr_loss
                + params.lambda_std * std_loss
                + params.lambda_cov * cov_loss
#                 + 25 * std_loss2
            )

            # Keep track of the elements of the loss
            dist_memory.append(repr_loss.item()* size_of_batch)
            var_memory.append(std_loss.item()* size_of_batch)
            var_memory2.append(std_loss2.item()* size_of_batch)
            cov_memory.append(cov_loss.item()* size_of_batch)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * size_of_batch
            samples += size_of_batch
                
#             if i == 1:
#                 break

        # Log
        logger.log({"repr_loss": np.mean(dist_memory),
        "std_loss": np.mean(var_memory),
        "std_loss2": np.mean(var_memory2),
        "cov_loss": np.mean(cov_memory),
        "global_loss": trainLoss / samples})

        scheduler.step(trainLoss / samples)
        trainTemplate = "TRAIN - epoch: {} train loss: {:.6f} learning rate: {:.6f}"
        print(trainTemplate.format(epoch + 1, (trainLoss / samples),
            (scheduler.optimizer.param_groups[0]['lr'])))
        loss_memory.append(trainLoss / samples)
        if scheduler.optimizer.param_groups[0]['lr'] < 1e-4 :
            break
        
        # From time to time let's see wehat that models output on validation data
        if (epoch+1)%params.test_interval==0:
            testing_model(t_loader, v_loader, mymodel, logger)

    testing_model(t_loader, v_loader, mymodel, logger)

    #end
    return loss_memory, cov_memory, var_memory, var_memory2, dist_memory, encodded_validations



if __name__ == "__main__":
    # test training process with dumy data
    training_model()