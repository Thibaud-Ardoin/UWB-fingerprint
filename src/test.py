"""
    Test.py
    this files links to the different evaluation precesses and post processing of the data
    Classification, clustering algorithms
"""

import numpy as np
import time

import torch
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from sklearn.metrics import classification_report, confusion_matrix

import params


def encode_data(mymodel, dataloader):
    mymodel.eval()
    def loop_on_loader(loader):
        loader.dataset.testset = True
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


def reid_evaluation_test2train_NN(test_embeddings, train_embeddings, test_labels, train_labels, logger):
    mat = distance_matrix(test_embeddings, train_embeddings)

    # Nearest Neighbor accuracy
    correct_classification = 0
    wrong_classification = 0
    nearest10_amt = 0
    for i in range(len(mat)):
        nearest_10id = np.argpartition(mat[i], 11)[1:11]
        nearest_id = nearest_10id[0]
        if train_labels[nearest_id] == test_labels[i]:
            correct_classification +=1
        else :
            wrong_classification +=1
        
        nearest10_amt += len(np.where(np.array(train_labels)[nearest_10id] == test_labels[i])[0])

    logger.log({
        "NN test2train": 100*correct_classification/(correct_classification+wrong_classification),
        "NN test2train 10th mean": 100*nearest10_amt/(10*len(mat)),
    })

    if params.verbose:
        print("> Accuracy for positive classification according to the nearest neighbor from test in training set ")
        print(100*correct_classification/(correct_classification+wrong_classification), "%")
        
        print("> Mean percentage of same class points in the 10th nearest elmts from test in training set ")
        print(100*nearest10_amt/(10*len(mat)), "%")
    





def reid_evaluation(embeddings, labels, logger):    
    mat = distance_matrix(embeddings, embeddings)
    
    
    same_distances = []
    diff_distances = []

    dev_ind = [np.where(np.array(labels)==i)[0] for i in range(params.num_dev)]
    for i in range(params.num_dev):
        for j in range(i, params.num_dev):
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
    area = np.trapz(roc_value[:, 1], roc_value[:, 0])

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


    logger.log_curve(roc_value[:, :2], title="Clustering Metric", column_names=["Positive", "Negative"])
    logger.log({"Positive clustering at error rate 0.1%": 100*ta_accuracy[0],
            "Positive clustering at error rate 1%": 100*ta_accuracy[1],
            "Positive clustering at error rate 10%": 100*ta_accuracy[2],
            "Trapz": area,
            "NN classification on test data": 100*correct_classification/(correct_classification+wrong_classification)})

    
    if params.verbose:
        print("Percentage of each label by value", 100*np.histogram(labels, bins=len(set(labels)))[0]/len(labels))

        print("MaxMin of same:", max(same_distances), min(same_distances))
        print("MaxMin of diff:", max(diff_distances), min(diff_distances))

        print("> Amount of positive clustering when False accuracy reaches 0,1% 1% and 10%")
        print(list(100*np.array(ta_accuracy)), "%")

        print("> Area size under the clustering curve")
        print(area)

        print("> Accuracy for positive classification according to the nearest neighbor ")
        print(100*correct_classification/(correct_classification+wrong_classification), "%")
        
        print("> Mean percentage of same class points in the 10th nearest elmts ")
        print(100*nearest10_amt/(10*len(mat)), "%")
    
    if params.plotting:                    
        plt.plot(roc_value[:, 0], roc_value[:, 1])
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.show()

def evaluate_Kmeans(test_set, test_labels, logger):
    data_size = len(test_set)
    kmeans = KMeans(n_clusters=params.num_dev, random_state=0, n_init="auto").fit(test_set)
    dev_ids = [np.where(test_labels == i) for i in range(params.num_dev)]
    accuracy = 0
    for ids in dev_ids :
        hist = np.histogram(kmeans.labels_[ids], bins=params.num_dev)
        accuracy += np.max(hist[0])

    accuracy = 100*accuracy / data_size

    logger.log({"Kmeans cluster score %": accuracy})

    if params.verbose:
        print(" > Accuracy from Kmeans on clustering the test set: ", accuracy)

def accuracy_test(model, encoded_test, labels_test, logger):
    encoded = torch.Tensor(encoded_test).to(params.device)
    labels = torch.Tensor(labels_test).to(params.device)
    output = model.classify(encoded)
    predicted = torch.argmax(output, dim=1)
    target_count = labels.size(0)
    correct_val = (labels == predicted).sum().item()
    val_acc = 100 * correct_val / target_count
    
    logger.log({"Accuracy for dev classification on test data %": val_acc})
    if params.verbose:
        print("Accuracy for dev classification on test data %", val_acc)
        print(classification_report(labels.cpu(), predicted.cpu()))
        print(confusion_matrix(labels.cpu(), predicted.cpu()))

        

def testing_model(training_loaders, validation_loader, model, logger):
    start_time = time.time()

    encoded_train, labels_train = encode_data(model, training_loaders)
    encoded_test, labels_test = encode_data(model, validation_loader)
    enc_time = time.time()
    print("[Test]: time for encodding", enc_time - start_time)

    logger.log_scatter(encoded_train, labels_train, title="train_data")
    logger.log_scatter(encoded_test, labels_test, title="test_data")
    scatter_time = time.time()
    print("[Test]: time for plottingscatter", scatter_time - enc_time)

    # Selecting a subset of size params.data_test_rate for reid evalution
    mask = np.full(len(encoded_test), False)
    mask[:int(len(encoded_test)*params.data_test_rate)] = True
    np.random.shuffle(mask)
    reid_evaluation(np.array(encoded_test)[mask], np.array(labels_test)[mask], logger)
    reid_time = time.time()
    print("[Test]: time for reid on test data", reid_time - scatter_time)

    mask2 = np.full(len(labels_train), False)
    mask2[:int(len(labels_train)*params.data_test_rate*params.data_test_rate)] = True
    np.random.shuffle(mask2)
    reid_evaluation_test2train_NN(np.array(encoded_test)[mask], np.array(encoded_train)[mask2], np.array(labels_test)[mask], np.array(labels_train)[mask2], logger)
    reid2_time = time.time()
    print("[Test]: time for test 2 train reid", reid2_time - reid_time)

    evaluate_Kmeans(np.array(encoded_test)[mask], np.array(labels_test)[mask], logger)
    kmeans_time = time.time()
    print("[Test]: time for test K_means evaluation", kmeans_time - reid2_time)

    if params.loss in ["crossentropy", "triplet+crossentropy"]:
        accuracy_test(model, encoded_test, labels_test, logger)
    

    logger.step_test()


if __name__ == "__main__":
    logger = None
    # Dummy data
    test_set = np.random.rand(1000, 64)
    test_labels = np.random.rand(1000)*13
    test_labels = np.floor(test_labels)

    evaluate_Kmeans(test_set, test_labels, logger)
