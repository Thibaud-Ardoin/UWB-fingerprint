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

from scipy.spatial import distance_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from optimizer import Optimizer, AdvOptimizer
import params
from test import testing_model

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def accuracy(target, prediction):
    predicted = torch.argmax(prediction, dim=1)
    target_count = target.size(0)
    correct_val = (target == predicted).sum().item()
    val_acc = 100 * correct_val / target_count
    return val_acc


def training_model(t_loader, v_loader, mymodel, logger):
    encodded_validations = []

    # Needs reorganisation, encapsulation PLS
    if params.model_name == "advCNN1":
        optimizer = AdvOptimizer(mymodel)
    else:
        optimizer = Optimizer(mymodel)

    # Actually all this loss mess needs to be encapsulated too !
    if params.loss.startswith("triplet"):
        tripletLoss = nn.TripletMarginLoss(margin=params.triplet_mmargin, p=2, reduce=True, reduction="mean")
    elif params.loss in ["adversarial", "crossentropy"] :
        CEloss = nn.CrossEntropyLoss()


    if not params.flat_data :
        min_size = None
        for pos in range(len(t_loader)) :
            for dev in range(len(t_loader[pos])):
                if min_size is None or min_size > len(t_loader[pos][dev].dataset):
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
        loss_memory, cov_memory, dist_memory, var_memory, var_memory2, pos_accuracy, dev_accuracy = [], [], [], [], [], [], []

        # loop over the current batch of data
        if params.flat_data:
            for i, data in enumerate(t_loader):

                x, y = data

                if params.loss == "adversarial":

                    dev_pred, pos_pred = mymodel(x)

                    devLoss = CEloss(dev_pred.double(), y[:,0])
                    posLoss = CEloss(pos_pred.double(), y[:,1])

                    encLoss = (
                        devLoss - posLoss
                    )

                    # Backprop
                    optimizer.zero_grad()

                    devLoss.backward(retain_graph=True)
                    posLoss.backward(retain_graph=True)
                    encLoss.backward()

                    optimizer.step()

                    devAcc = accuracy(y[:,0], dev_pred)
                    posAcc = accuracy(y[:,1], pos_pred)

                    var_memory.append(devLoss.item())
                    cov_memory.append(posLoss.item())
                    pos_accuracy.append(posAcc)
                    dev_accuracy.append(devAcc)
                    # var_memory2.append(trip_loss.item())

                    size_of_batch = x.size(0)

                    trainLoss += encLoss.item() * size_of_batch
                    samples += size_of_batch

                elif params.loss == "crossentropy":

                    dev_pred = mymodel(x)

                    devLoss = CEloss(dev_pred.double(), y[:,0])

                    # Backprop
                    optimizer.zero_grad()
                    devLoss.backward()
                    optimizer.step()

                    devAcc = accuracy(y[:,0], dev_pred)

                    var_memory.append(devLoss.item())
                    dev_accuracy.append(devAcc)

                    size_of_batch = x.size(0)

                    trainLoss += devLoss.item() * size_of_batch
                    samples += size_of_batch

        else :
            # Data divided in multi class
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


                if params.loss == "triplet3":
                    # Triplet with hard negative exemple mining
                    p1 = np.random.choice(pos_amt)
                    p2 = np.random.choice([p for p in range(pos_amt) if p!=p1])

                    # Two data batches, with same device and different position
                    batchX1 = [next(iter(t_loader[dev][p1]))[0] for dev in range(len(t_loader))]
                    batchX2 = [next(iter(t_loader[dev][p2]))[0] for dev in range(len(t_loader))]

                    x1 = [mymodel(batchX1[dev]) for dev in range(len(t_loader))]
                    x2 = [mymodel(batchX2[dev]) for dev in range(len(t_loader))]

                    # Covariance
                    x = torch.cat(x1)
                    y = torch.cat(x2)                                
                    x = x - x.mean(dim=0)
                    y = y - y.mean(dim=0)

                    cov_x = (x.T @ x) / (params.batch_size - 1)
                    cov_y = (y.T @ y) / (params.batch_size - 1)
                    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                        mymodel.embedding_size
                    ) + off_diagonal(cov_y).pow_(2).sum().div(mymodel.embedding_size)


                    distance_mat = distance_matrix(
                        np.array([torch.flatten(xx1).cpu().detach().numpy() for xx1 in x1]),
                        np.array([torch.flatten(xx2).cpu().detach().numpy() for xx2 in x2])
                    )
                    
                    # Triplet loss
                    trip_loss = 0
                    for dev1 in range(params.num_dev):
                        if epoch > 100 and np.random.rand() > 0.05:
                            neighbors = np.argpartition(distance_mat[dev1], 2)
                            dev2 = neighbors[1]
                            if dev2==dev1:
                                dev2 = neighbors[0]
                        else :
                            dev2 = np.random.choice([d for d in range(params.num_dev) if d!=dev1])

                        anchor = x1[dev1]         # P1, D1 
                        positive = x2[dev1]       # P2, D1
                        negative = x1[dev2]       # P1, D2

                        trip_loss += tripletLoss(anchor, positive, negative)

                    loss = (
                        params.lambda_triplet * trip_loss / params.num_dev +
                        params.lambda_cov * cov_loss
                    )

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    var_memory2.append(trip_loss.item()) # Actually triplet loss hey
                    cov_memory.append(cov_loss.item()) # Actually triplet loss hey

                    size_of_batch = anchor.size(0)

                    trainLoss += loss.item() * size_of_batch
                    samples += size_of_batch

                ####### SETUP WITH TWO RANDOM POSITIONS EVERY TIME


                elif params.loss == "triplet2":
                    # Just the triplet loss but compiled on all devices 
                    p1 = np.random.choice(pos_amt)
                    p2 = np.random.choice([p for p in range(pos_amt) if p!=p1])
                    
                    # Two data batches, with same device and different position
                    batchX1 = [next(iter(t_loader[dev][p1]))[0] for dev in range(len(t_loader))]
                    batchX2 = [next(iter(t_loader[dev][p2]))[0] for dev in range(len(t_loader))]

                    x1 = [mymodel(batchX1[dev]) for dev in range(len(t_loader))]
                    x2 = [mymodel(batchX2[dev]) for dev in range(len(t_loader))]
                    
                    # Triplet loss
                    trip_loss = 0
                    for dev1 in range(params.num_dev):
                        dev2 = np.random.choice([d for d in range(params.num_dev) if d!=dev1])

                        anchor = x1[dev1]         # P1, D1 
                        positive = x2[dev1]       # P2, D1
                        negative = x1[dev2]       # P1, D2

                        trip_loss += tripletLoss(anchor, positive, negative)

                    loss = (
                        params.lambda_triplet * trip_loss
                    )

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    var_memory2.append(trip_loss.item()) # Actually triplet loss hey

                    size_of_batch = anchor.size(0)

                    trainLoss += loss.item() * size_of_batch
                    samples += size_of_batch

                ####### SETUP WITH TWO RANDOM POSITIONS EVERY TIME



                elif params.loss == "triplet":
                    # Triplet loss + VICreg loss
                    p1 = np.random.choice(pos_amt)
                    p2 = np.random.choice([p for p in range(pos_amt) if p!=p1])
                    
                    # Two data batches, with same device and different position
                    batchX1 = [next(iter(t_loader[dev][p1]))[0] for dev in range(len(t_loader))]
                    batchX2 = [next(iter(t_loader[dev][p2]))[0] for dev in range(len(t_loader))]

                    x1 = [mymodel(batchX1[dev]) for dev in range(len(t_loader))]
                    x2 = [mymodel(batchX2[dev]) for dev in range(len(t_loader))]
                    
        #             repr_loss = F.mse_loss(x, y)
                    repr_loss = torch.stack([F.mse_loss(x1[dev], x2[dev]) for dev in range(params.num_dev)]).mean()

                    # Covariance
                    x = torch.cat(x1)
                    y = torch.cat(x2)                                
                    x = x - x.mean(dim=0)
                    y = y - y.mean(dim=0)

                    cov_x = (x.T @ x) / (params.batch_size - 1)
                    cov_y = (y.T @ y) / (params.batch_size - 1)
                    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                        mymodel.embedding_size
                    ) + off_diagonal(cov_y).pow_(2).sum().div(mymodel.embedding_size)

                    # Variance
                    xt1 = torch.stack([x1[dev] for dev in range(len(t_loader))],  dim=0)
                    xt2 = torch.stack([x2[dev] for dev in range(len(t_loader))],  dim=0)
                    std_x = torch.sqrt(xt1.var(dim=0) + 0.0001)
                    std_y = torch.sqrt(xt2.var(dim=0) + 0.0001)
                    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

                    
                    # Triplet loss
                    dev1 = np.random.choice(params.num_dev)
                    dev2 = np.random.choice([d for d in range(params.num_dev) if d!=dev1])

                    anchor = x1[dev1]         # P1, D1 
                    positive = x2[dev1]       # P2, D1
                    negative = x1[dev2]       # P1, D2



                    trip_loss = tripletLoss(anchor, positive, negative)

                    loss = (
                        params.lambda_triplet * trip_loss +
                        params.lambda_distance * repr_loss + 
                        params.lambda_cov * cov_loss + 
                        params.lambda_std * std_loss
                    )

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    dist_memory.append(repr_loss.item())
                    cov_memory.append(cov_loss.item())
                    var_memory.append(std_loss.item()) # Actually triplet loss hey
                    var_memory2.append(trip_loss.item()) # Actually triplet loss hey

                    size_of_batch = anchor.size(0)

                    trainLoss += loss.item() * size_of_batch
                    samples += size_of_batch

                ####### SETUP WITH TWO RANDOM POSITIONS EVERY TIME
                
                elif params.loss == "vicreg":
                    p1 = np.random.choice(pos_amt)
                    p2 = np.random.choice([p for p in range(pos_amt) if p!=p1])
                    
                        # Two data batches, with same device and different position
                    batchX1 = [next(iter(t_loader[dev][p1]))[0] for dev in range(len(t_loader))]
                    batchX2 = [next(iter(t_loader[dev][p2]))[0] for dev in range(len(t_loader))]

                    c1 = torch.concatenate(batchX1)
                    c2 = torch.concatenate(batchX2)
                    z1 = mymodel(c1)
                    z2 = mymodel(c2)

                    x1 = [z1[dev*params.batch_size:(dev+1)*params.batch_size] for dev in range(len(t_loader))]
                    x2 = [z2[dev*params.batch_size:(dev+1)*params.batch_size] for dev in range(len(t_loader))]

                    # Same as z1, z2 ?
                    x = torch.cat(x1)
                    y = torch.cat(x2)
                    
        #             repr_loss = F.mse_loss(x, y)
                    repr_loss = torch.stack([F.mse_loss(x1[dev], x2[dev]) for dev in range(params.num_dev)]).sum()
                    
                    x = x - x.mean(dim=0)
                    y = y - y.mean(dim=0)

                    size_of_batch = x.size(0)

                    
        #             xt1 = torch.cat([x1[dev][None] for dev in range(num_dev)],  dim=0)
        #             xt2 = torch.cat([x2[dev][None] for dev in range(num_dev)],  dim=0)
                    xt1 = torch.stack([x1[dev] for dev in range(len(t_loader))],  dim=0)
                    xt2 = torch.stack([x2[dev] for dev in range(len(t_loader))],  dim=0)
                    
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
                        mymodel.embedding_size
                    ) + off_diagonal(cov_y).pow_(2).sum().div(mymodel.embedding_size)
                    # cov_loss = F.relu(1 - cov_loss)

                    # Paper's parameters 25, 25, 1
                    loss = (
                        params.lambda_distance * repr_loss
                        + params.lambda_std * std_loss
                        + params.lambda_cov * cov_loss
        #                 + 25 * std_loss2
                    ) / (params.lambda_distance + params.lambda_std + params.lambda_cov)

                    # Keep track of the elements of the loss
                    dist_memory.append(repr_loss.item())
                    var_memory.append(std_loss.item())
                    var_memory2.append(std_loss2.item())
                    cov_memory.append(cov_loss.item())

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    trainLoss += loss.item() * size_of_batch
                    samples += size_of_batch
                        
        #             if i == 1:
        #                 break

        # Log
        
        if params.loss=="crossentropy":
            logger.log({
            "Dev class loss": np.mean(var_memory),
            "Dev class accuracy": np.mean(dev_accuracy),
            "Encoder loss": trainLoss / samples,
            "learning rate": optimizer.get_lr()})
            logger.step_epoch()
        if params.loss=="adversarial":
            logger.log({
            "Dev class loss": np.mean(var_memory),
            "Pos class loss": np.mean(cov_memory),
            "Pos class accuracy": np.mean(pos_accuracy), 
            "Dev class accuracy": np.mean(dev_accuracy),
            "Encoder loss": trainLoss / samples,
            "learning rate": optimizer.get_lr()})
            logger.step_epoch()
        elif params.loss=="triplet3":
            logger.log({
            "triploss": np.mean(var_memory2),
            "cov_loss": np.mean(cov_memory),
            "global_loss": trainLoss / samples,
            "learning rate": optimizer.get_lr()})
            logger.step_epoch()
        elif params.loss=="triplet2":
            logger.log({
            "triploss": np.mean(var_memory2),
            "global_loss": trainLoss / samples,
            "learning rate": optimizer.get_lr()})
            logger.step_epoch()
        elif params.loss=="triplet":
            logger.log({
            "repr_loss": np.mean(dist_memory),
            "cov_loss": np.mean(cov_memory),
            "std_loss": np.mean(var_memory),
            "triploss": np.mean(var_memory2),
            "global_loss": trainLoss / samples,
            "learning rate": optimizer.get_lr()})
            logger.step_epoch()

        elif params.loss=="vicreg":
            logger.log({"repr_loss": np.mean(dist_memory),
            "std_loss": np.mean(var_memory),
            "std_loss2": np.mean(var_memory2),
            "cov_loss": np.mean(cov_memory),
            "global_loss": trainLoss / samples,
            "learning rate": optimizer.get_lr()})
            logger.step_epoch()


        trainTemplate = "TRAIN - epoch: {} train loss: {:.6f} learning rate: {:.6f}"
        print(trainTemplate.format(epoch + 1, (trainLoss / samples),
            (optimizer.get_lr())))
        loss_memory.append(trainLoss / samples)

        # Optimizer
        optimizer.epoch_routine(trainLoss / samples)
        if optimizer.early_stopping() :
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