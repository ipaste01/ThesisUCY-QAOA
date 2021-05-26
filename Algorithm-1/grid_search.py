# -*- coding: utf-8 -*-
"""
Updated on Thu May 26 16:31:50 2021

@author: Ioannis Pastellas
"""

#import math tools
import numpy as np

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

def generate_grid_numpy(p):
  
   sample = list()
   step_size = 0.5
   a_gamma         = np.arange(0, 2*np.pi, step_size)
   a_beta          = np.arange(0, np.pi, step_size)
   
   point_g = []
   point_b = []
   
   for i in range(p):
      point_g.append(a_gamma)
      point_b.append(a_beta)
      
   point = point_g + point_b
   sample = np.meshgrid(point) 
    
   return sample


def generate_grid_2(p,step):
  
   sample = list()
   step_size = step
   a_gamma         = np.arange(0, 2*np.pi, step_size)
   a_beta          = np.arange(0, np.pi, step_size)
   
   point_g = []
   point_b = []
   smaple = []
   for i in (a_gamma):
       for j in (a_beta):
                   sample.append([i,j])   
  
   print(sample) 
   return sample



