import numpy as np 
from MORALS.systems.system import BaseSystem

class Sphere2(BaseSystem):
    def __init__(self,**kwargs):
        self.name = "sphere2"
        self.state_bounds = np.array([[-2.5, 2.5]]*3)
    
    # def get_true_bounds(self):
    #     return NotImplementedError
    
    # def get_bounds(self):
    #     return NotImplementedError