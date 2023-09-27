"""
    Test.py
    this files links to the different evaluation precesses and post processing of the data
    Classification, clustering algorithms
"""

import params


def encode_data(mymodel, dataloader):
    mymodel.eval()
    def loop_on_loader(loader):
        loader.dataset.augmentation =False
        encs, labs = [], []
        for i, (batchX, batchY) in enumerate(loader):
            # Compute encoded version of the data by our embedding model
            encs = encs + mymodel.encoder(batchX).tolist()
            # Gather device labels accordingly (eventually randomly enumerated)
            labs = labs + batchY[:, 0].tolist()
        return encs, labs
    
    # Split differently according to the dataloader organisation
    myencodings, mylabels = [], []
    if type(dataloader) == list :
        for pos_loader in dataloader :
            for dev_loader in pos_loader :  
                encs, labs = loop_on_loader(dev_loader)
                myencodings = myencodings + encs
                mylabels = mylabels + labs
    else :    
        myencodings, mylabels = loop_on_loader(dataloader)
        
    return myencodings, mylabels





def reid_evaluation(embeddings, labels, logger):    
    mat = distance_matrix(embeddings, embeddings)
    
    
    same_distances = []
    diff_distances = []

    dev_ind = [np.where(np.array(labels)==i)[0] for i in range(num_dev)]
    for i in range(num_dev):
        for j in range(i, num_dev):
            sub_mat = mat[dev_ind[i]].T
            subsub_mat = sub_mat[dev_ind[j]]
            for ind_i in range(len(subsub_mat)):
                for ind_j in range(ind_i+1, len(subsub_mat[0])):
                    d = subsub_mat[ind_i][ind_j]
                    if i==j :
                        same_distances.append(d)
                    else :
                        diff_distances.append(d)

    roc_value = []
    th_steps = 50
    max_dist = max(diff_distances)
    min_dist = min(same_distances)

    for th in range(th_steps+1):
        t = ((th+0.5)/th_steps)
        t = t*t
        threshold = t*max_dist + (1 - t)*min_dist
#         threshold = threshold * threshold
        ta = len(np.where(np.array(same_distances) < threshold)[0])/len(same_distances)
        fa = len(np.where(np.array(diff_distances) < threshold)[0])/len(diff_distances)
        roc_value.append([fa, ta, threshold])        
    roc_value = np.array(roc_value) 

    # Get accuracy when fa=0.1% , 1%, 10%
    targets_fa = [0.001, 0.01, 0.1]
    current_target_fa_id = 0
    target_th = []
    ta_accuracy =[]
    for i in range(len(roc_value)):
        if roc_value[i][0] >= targets_fa[current_target_fa_id]:
            target_th.append(roc_value[i][2])
            ta_accuracy.append(roc_value[i][1])
            current_target_fa_id += 1
            if current_target_fa_id >= len(targets_fa):
                break
        

    # Nearest Neighbor accuracy
    correct_classification = 0
    wrong_classification = 0
    nearest10_amt = 0
    for i in range(len(mat)):
        nearest_id = np.argpartition(mat[i], 2)[1]
        nearest_10id = np.argpartition(mat[i], 11)[1:11]
        if labels[nearest_id] == labels[i]:
            correct_classification +=1
        else :
            wrong_classification +=1
        
        nearest10_amt += len(np.where(np.array(labels)[nearest_10id] == labels[i])[0])


    logger.log_curve(roc_value, title="Clustering Metric")
    logger.log()

    
    if params.verbose:
        print("Percentage of each label by value", 100*np.histogram(labels, bins=len(set(labels)))[0]/len(labels))

        print("MaxMin of same:", max(same_distances), min(same_distances))
        print("MaxMin of diff:", max(diff_distances), min(diff_distances))

        print("> Amount of positive clustering when False accuracy reaches 0,1% 1% and 10%")
        print(list(100*np.array(ta_accuracy)), "%")

        print("> Accuracy for positive classification according to the nearest neighbor ")
        print(100*correct_classification/(correct_classification+wrong_classification), "%")
        
        print("> Mean percentage of same class points in the 10th nearest elmts ")
        print(100*nearest10_amt/(10*len(mat)), "%")
    
    if params.plotting:                    
        plt.plot(roc_value[:, 0], roc_value[:, 1])
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.show()
        
# reid_evaluation(encoded_test[:1000], labels_test[:1000])

def testing_model(training_loaders, validation_loader, model, logger ):
    encoded_train, labels_train = encode_data(model, training_loaders)
    encoded_test, labels_test = encode_data(model, validation_loader)

    logger.log_scatter(encoded_train, labels_train, title="train_data")
    logger.log_scatter(encoded_test, labels_test, title="test_data")

    reid_evaluation(encoded_test, labels_test)


