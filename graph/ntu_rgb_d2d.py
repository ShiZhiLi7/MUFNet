
import numpy as np

from graph import tools

num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 0), (0, 14), (0, 15), (14, 16), (15, 17), (1, 5), (1, 2),
                    (2, 3), (3, 4),  (5, 6), (6, 7), (1, 8), (1, 11),
                    (8, 9), (9, 10), (11, 12), (12, 13)]
inward = [(i, j)for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
if __name__ == '__main__':
    graph = Graph()
    A = graph.A
    '''
     I In Out (3,25,25)
  '''
    print(A[1]+A[2])