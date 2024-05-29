import torch
import os
import numpy

from main_visualizer import load_data
from sklearn.neighbors import NearestNeighbors
from main_data_distribution_visualization import get_configuration


quantity_classes = 10
size_embedding = 8
quantity_neighbors = 10
quantity_samples_for_each_class = 10


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def extrapolate_n_for_each_class(x, y, n):
    result = []
    class_counter = []
    for j in range(quantity_classes):
        class_counter.append(0)

    new_x, new_y = unison_shuffled_copies(x, y)
    for i in range(len(new_x)):
        for j in range(quantity_classes):
            if new_y[i] == j and class_counter[j] < n:
                result.append(new_x[i])
                class_counter[j] += 1
        
        check_all_full = True
        for j in range(quantity_classes):
            if class_counter[j] != n:
                check_all_full = False
                break
        
        if check_all_full:
            break

    return result


def get_sample_class(x, y, sample):
    index = numpy.where(x == sample)[0][0]
    return y[index]


if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    configuration = get_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}...".format(device))
    sequence_length = configuration.default_num_hiddens * size_embedding * size_embedding
    x, y = load_data(device)
    knn = NearestNeighbors(n_neighbors=quantity_neighbors)
    knn.fit(x)
    points_classes = extrapolate_n_for_each_class(x, y, quantity_samples_for_each_class)
    accuracies_same_class = {}
    for point in points_classes:
        point_y = get_sample_class(x, y, point)
        neighbors = knn.kneighbors(point.reshape(1, -1), return_distance=False)
        quantity_same_class = 0
        for neighbor_index in neighbors[0]:
            if point_y == y[neighbor_index]:
                quantity_same_class += 1
        
        if point_y in accuracies_same_class:
            accuracies_same_class[point_y] += (quantity_same_class/quantity_neighbors)
        
        else:
            accuracies_same_class[point_y] = (quantity_same_class/quantity_neighbors)
        
        #print("For a generic point of class {} there are {} neighbors on {} samples considered.".format(point_y, quantity_same_class, quantity_neighbors))
    
    for key in accuracies_same_class.keys():
        print("For the class {} the accuracy of the neighborhood of size {} considering {} samples is {}%.".format(key, quantity_neighbors, quantity_samples_for_each_class, int((accuracies_same_class[key]/quantity_samples_for_each_class)*100)))
