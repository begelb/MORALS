import numpy as np 
from MORALS.systems.system import BaseSystem

class Sphere(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "sphere"
        self.state_bounds = np.array([[-1, 1]]*4)
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError