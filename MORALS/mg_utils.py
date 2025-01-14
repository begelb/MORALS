import numpy as np 

from MORALS.systems.utils import get_system
from collections import defaultdict
from MORALS.grid import Grid

import os

class MorseGraphOutputProcessor:
    def __init__(self, config, output_dir):
        mg_roa_fname = os.path.join(output_dir, 'MG_RoA_.csv')
        mg_att_fname = os.path.join(output_dir, 'MG_attractors.txt')
        mg_fname = os.path.join(output_dir, 'MG')
        # mg_roa_fname = os.path.join(config['output_dir'], 'MG_RoA_.csv')
        # mg_att_fname = os.path.join(config['output_dir'], 'MG_attractors.txt')
        # mg_fname = os.path.join(config['output_dir'], 'MG')

        self.dims = config['low_dims']

        # Check if the file exists
        if not os.path.exists(mg_roa_fname):
            raise FileNotFoundError("Morse Graph RoA file does not exist at: " + output_dir)
        with open(mg_roa_fname, 'r') as f:
            lines = f.readlines()
            # Find indices where the first character is an alphabet
            self.indices = []
            for i, line in enumerate(lines):
                if line[0].isalpha():
                    self.indices.append(i)
            first_num_line = np.array(lines[self.indices[0]+1].split(',')).astype(np.float32)
            # Extract only the first self.dim numerical values in the file as box_size
            # Previously self.box_size was equal to the entire line, and then np.prod(self.box_size) was only correct by accident when the lower bounds were -1 and the upper bounds were 1
            # To do: test for dim > 2
            self.box_size = first_num_line[:self.dims]
            self.lower_bounds = first_num_line[self.dims:2*self.dims].tolist()
            self.upper_bounds = first_num_line[2*self.dims:3*self.dims].tolist()
            print('Box size: ', self.box_size)
            print('Lower bounds: ', self.lower_bounds)
            print('Upper bounds: ', self.upper_bounds)

            self.morse_nodes_data = np.vstack([np.array(line.split(',')).astype(np.float32) for line in lines[self.indices[1]+1:self.indices[2]]])
            if len(self.indices) >= 2:
                self.attractor_nodes_data = np.vstack([np.array(line.split(',')).astype(np.float32) for line in lines[self.indices[2]+1:]])
            else:
                line = lines[self.indices[2]+1]
                self.attractor_nodes_data = np.array(line.split(',').astype(np.float32))

        self.morse_nodes = np.unique(self.morse_nodes_data[:, 1])

        self.corner_points = {}
        for i in range(self.morse_nodes_data.shape[0]):
            self.corner_points[int(self.morse_nodes_data[i, 0])] = int(self.morse_nodes_data[i, 1])
        for i in range(self.attractor_nodes_data.shape[0]):
            self.corner_points[int(self.attractor_nodes_data[i, 0])] = int(self.attractor_nodes_data[i, 1])

        if not os.path.exists(mg_att_fname):
            raise FileNotFoundError("Morse Graph attractors file does not exist at: " + output_dir)
        
        self.found_attractors = -1
        with open(mg_att_fname, 'r') as f:
            line = f.readline()
            # Obtain the last number after a comma
            self.found_attractors = int(line.split(",")[-1])
            # Find the numbers enclosed in square brackets
            self.attractor_nodes = np.array([int(x) for x in line.split("[")[1].split("]")[0].split(",")])

        if not os.path.exists(mg_fname):
            raise FileNotFoundError("Morse Graph file does not exist at: " + output_dir)
        
        self.incoming_edges = defaultdict(list)
        self.outgoing_edges = defaultdict(list)
        with open(mg_fname, 'r') as f:
            # Check for lines of the form a -> b;
            for line in f.readlines():
                if line.find("->") != -1:
                    a = int(line.split("->")[0].strip())
                    b = int(line.split("->")[1].split(";")[0].strip())
                    self.outgoing_edges[a].append(b)
                    self.incoming_edges[b].append(a)

        # lower_bounds = [-1.]*self.dims
        # upper_bounds = [1.]*self.dims
    
        latent_space_area = np.prod(np.array(self.upper_bounds) - np.array(self.lower_bounds))
        print('self.box_size', self.box_size)
        box_area = np.prod(self.box_size)
        print('latent_space_area', latent_space_area)
        print('box_area', box_area)
        print('np.log2(latent_space_area/box_area)', np.log2(latent_space_area/box_area))
        subdivisions = np.log2(latent_space_area/box_area)
        print('round(subdivisions)', round(subdivisions))
        self.grid = Grid(self.lower_bounds, self.upper_bounds, int(round(subdivisions)))

    def check_in_bounds(self, point):
        in_bounds_list = []
        for d in range(self.dims):
            in_bounds = self.lower_bounds[d] <= point[d] <= self.upper_bounds[d]
            in_bounds_list.append(in_bounds)
        return all(in_bounds_list)

    def get_num_attractors(self):
        return self.found_attractors
    
    def get_corner_points_of_attractor(self, id):
        # Get the attractor nodes
        attractor_nodes = self.attractor_nodes_data[self.attractor_nodes_data[:, 1] == id]
        return attractor_nodes[:, 2:]        

    def get_corner_points_of_morse_node(self, id):
        morse_node_nodes = self.morse_nodes_data[self.morse_nodes_data[:, 1] == id]
        return morse_node_nodes[:, 2:]
    
    def which_morse_node(self, point):
        assert point.shape[0] == self.dims
        found = self.corner_points[self.grid.point2indexCMGDB(point)]
        return found