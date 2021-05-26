# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:16:04 2021

@author: Ioannis Pastellas
"""

#import math tools
import numpy as np
import random as rd
import itertools
from random import randrange

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

def coefficients(V,x,length):
  while len(x) < length:
      x.insert(0,0)
  
  # find coefficents
  K = []
  c = [0]*V
  z = 0
  for i in range(V):
   for j in range(i+1,V):
      if x[z] == 1 :
         c[i] = 1
         c[j] = 1
         K.append(c)
      z = z + 1
      c = [0]*V
  '''for edge in E:
    c[edge[0]] = 1
    c[edge[1]] = 1
    K.append(c)
    c = [0]*len(V)'''
  return K 

def vector_add(x,y):
    if (len(x)== 0 and len(y)==0 ):
     return [];
    elif len(x) == 0:
     return y;
    elif len(y) == 0: return x;
   
    d = []
    
 
    for i in range(len(x)):
      d.append(x[i]+y[i])
    
    return d

def hamming_distance(x,y):
    d = 0
    if (len(x)!= len(y)):
     return 0;
 
    for i in range(len(x)):
      if (x[i] != y[i]):
        d = d + 1
    
    return d

def common_node(x,y):
    d = 0
    if (len(x)!= len(y)):
     return 0;
 
    for i in range(len(x)):
      if (x[i] == y[i]) and (x[i] == 1):
        return 1;
    
    return 0


def neighbour_coefficients(c,coefs):
    flag = False
    p = []
    q = []
    
    for coef in coefs:
        if (common_node(c,coef) == 1):
           if (flag == True) : #and (hamming_distance(coef,p)!=2) :
             q = coef
             coefs.remove(p)
             coefs.remove(q)
             return (p,q)
           else :
             p = coef
             flag = True
   
    return (p,q);


        

def validate_oracle(lenV,x,length):
  coefs = coefficients(lenV, x, length);  #finds the set of existing coefficients
  loop = True
  if (len(coefs) != lenV):
     return 0;                # no solution(must have n edges)
  s = coefs[randrange(len(coefs))]
  coefs.remove(s)
  
  
  while loop == True :
    loop = False
    (p,q ) = neighbour_coefficients(s, coefs)  # finds cp, cq 
    
    if (len(p) == 0 and len(q) == 0 ):
        return 0;                              # two coefficients not found
   
    s = vector_add(vector_add(s,p),q)

    for i in range(len(s)):
        if s[i] > 2:
           return 0;         # infeasible solution
    for i in range(len(s)):
        if s[i] == 1:
           loop = True       # we go to loop
    if loop==True:
       continue
       
    for i in range(len(s)):
        if s[i] == 0:
           return 0;         # closed disconnected cycle
    
    for i in range(len(s)):
      if s[i] != 2:
        return 0;
      else :return 1;       
    


def cost_function_C(x,weights,G):
    E = G.edges()
    V = G.nodes()
    
    if (len(x) != len(weights)):
        return np.nan;
    C = 0;
                
    for i in range(len(weights)):
          C = C + weights[i] * x[i]
         
          
        
    return C;

def flip_bits_indeces(x,z):
    comb = list((i,j,k,l) for ((i,_),(j,_),(k,_),(l,_)) in itertools.combinations(enumerate(x), z))
    
    return comb

def flip_bits(x,index,length):
  while len(x) < length:
    x.insert(0,0)
  y = [ x[i] for i in range(len(x))]
  for i in index:
      if (y[i] == 0):
          y[i] = 1;
      elif (y[i] == 1):
          y[i] = 0;
  return y;

def construct_cost_Hamiltonian(weights):
    
    H = [[0 for i in range(2**length)] for j in range(2**length)]
    for i in range(2**length):
       x  = [int(bit) for bit in list(list("{0:b}".format(i)))]
       while len(x) < length:
         x.insert(0,0)
       H[i][i] = cost_function_C(x, weights, G)
           
    return H;

def Hamiltonian_coloring(H):
    col = [[0 for i in range(2**length)] for j in range(2**length)]
    return col


def Hermitians(H,length):
    herm = []
    Hm = [[0 for i in range(2**length)] for j in range(2**length)]
    for i in range(2**length):
       for j in range(2**length):
           if (H[i][j] == 1):
               Hm[i][j] = 1
               Hm[j][i] = 1
               herm.append(Hm)
    return herm


    
def construct_constraint_Hamiltonian(length,G):
   H = [[0 for i in range(2**length)] for j in range(2**length)]
   feasible=[]
   gates_positions = []
   positions = []
   for num in range(2**length):
     x  = [int(bit) for bit in list(list("{0:b}".format(num)))]
     
     # Check for validate solution
     if (validate_oracle(len(G.nodes),x,length)==1):
        feasible.append(x)
        break
  
   
   #print(len(feasible))
   for x in feasible:
    positions = []
    for comb in flip_bits_indeces(x,4):
     y= flip_bits(x,comb,length)
     # Check for validate solution
     if (validate_oracle(len(G.nodes),y,length)==1):
       
       positions.append(comb)
       y_index = int("".join(str(k) for k in y), 2) 
       x_index = int("".join(str(k) for k in x), 2)
       H[x_index][y_index] = 1
       H[y_index][x_index] = 1
    gates_positions.append(positions)
   return (H,gates_positions,feasible);
    

#def prepare_cost():
#	return penalize_distances() + penalize_repeated_locations() + penalize_multiple_locations()

def read_edges():
    f = open("graph2.txt", "r")
    lines = f.readlines()
    E = []
    for line in lines:
        a = [float(x) for x in line.split()]
        if a[0] > a[1]:
            temp = a[0]
            a[0] = a[1]
            a[1] = temp
        E.append((int(a[0]),int(a[1]),a[2]))
    
       
    sorted(E,key=lambda tup: tup[0])
    return E

# Generating the butterfly graph with 5 nodes 
n     = 4
V     = np.arange(0,n,1)
#E     =[(0,1,1.0),(0,2,5.0),(1,2,5.0),(2,3,4.0),(4,3,1.0),(2,4,4.0)] 


E = read_edges()

def generate_weight_vector(G):
    length = int(( len(G.nodes)*(len(G.nodes)-1)/2))
    weights = [0] *length
    E = G.edges
        
    C = 0;
    z = 0;
    for i in range(len(V)):
        for j in range(i+1,len(V)):
         if (i,j) in E :
             weights[z] = G.get_edge_data(*(i,j))['weight']
         z = z + 1     
    for i in range(len(weights)):
      if (weights[i] == 0):  
        weights[i] =500; #float('inf');     # huge number      
    
    return weights


global G 
G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)


E = G.edges()
'''
x = [0] * int(( len(G.nodes)*(len(G.nodes)-1)/2))
z = 0;
for i in range(len(V)):
   for j in range(i+1,len(V)):
      if (i,j) in E :
         x[z] = 1
      z = z + 1
global length

length = int((len(G.nodes)*( len(G.nodes)-1)) / 2)'''
