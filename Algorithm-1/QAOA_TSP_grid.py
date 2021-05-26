
# -*- coding: utf-8 -*-
"""
Lat updated on  May 26 2021

@author: Ioannis Pastellas
"""


from qiskit import Aer, execute
from qiskit.aqua.components.initial_states import Custom

A = 500 #penalty
# Compute the value of the cost function MaxCut
def cost_function_C(x):
    C = 0
    for i in range(n):
        for j in range(n):
            for t in range(n-1):
              C = C+ adj[i][j] * x[t*n +i]*x[(t+1)*n + j]
          
    #penalties
    for t in range(0,n):
        s = 0
        for i in range(0,n):
            s = s + x[t*n+i]
        C = C + A*((1-s) **2)
        
     #penalties
    for i in range(0,n):
        s = 0
        for t in range(0,n):
            s = s + x[t*n+i]
        C = C + A*((1-s) **2)
          
    return C;

def objective(params):
    
    
    aqua_globals.random_seed = 10598
    quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   seed_simulator=aqua_globals.random_seed,
                                   seed_transpiler=aqua_globals.random_seed)
    qaoa_mes = QAOA(quantum_instance=quantum_instance,operator = qubitOp,p=n_layers,initial_point= params)   
    circuit2 = QuantumCircuit(n**2)
    circuit = qaoa_mes.construct_circuit(params)
    
    
    #create measurements on classical register
    reg = ClassicalRegister(n**2)
    circuit[0].add_register(reg)
    circuit[0].measure(range(n**2),range(n**2))
    
    
    # Execution of circuit(either simulation or on real quantum machines)
    backend      = Aer.get_backend("qasm_simulator")
    shots        = 1024

    simulate     = execute(circuit[0], backend=backend, shots=shots)
    QAOA_results = simulate.result()
    
    val = 0
    # Evaluate the data from the simulator
    counts = QAOA_results.get_counts()

    expectedCost       = 0
    maxCost       = [0,0]
    hist        = {}
    
    
    for sample in list(counts.keys()):
     
     # use sampled bit string x to compute C(x)
     x2  = [int(num) for num in list(sample)]
     tmp_eng   = cost_function_C(x2)
    
     # compute the expectation value and energy distribution
     expectedCost      = expectedCost     + counts[sample]*tmp_eng

    

    return expectedCost/shots; #/(shots);        
    #M1_sampled   = avr_C/shots


def read_graph(n):
    f = open("graph.txt", "r")
    lines = f.readlines()
    adje = [[0 for i in range(n)] for j in range(n)]
    for line in lines:
        a = [float(x) for x in line.split()]
        if a[0] > a[1]:
            temp = a[0]
            a[0] = a[1]
            a[1] = temp
        adje[int(a[0])][int(a[1])] = a[2]
        adje[int(a[1])][int(a[0])] = a[2]
    
    for i in range(n):
      for j in range(n):
         if (adje[i][j] == 0  ):
             adje[i][j] = 0 #attention
    #sorted(E,key=lambda tup: tup[0])
    return adje

#import all necessary libraries
import qiskit
qiskit.__qiskit_version__


# useful additional packages 

#import math tools
import numpy as np

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter


# importing Qiskit
from qiskit import Aer, IBMQ
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP,ADAM,P_BFGS
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from numpy import float64
from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram
#from qiskit.optimization.algorithms import CobylaOptimizer

from qiskit.aqua.operators.evolutions import Trotter,Suzuki, EvolvedOp,MatrixEvolution, PauliTrotterEvolution
from qiskit.quantum_info.operators import Operator
from qiskit.aqua.operators import OperatorBase,PrimitiveOp
from qiskit.aqua.operators.list_ops import SummedOp
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua import aqua_globals
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.optimization.applications.ising import max_cut, tsp
from qiskit.optimization import QuadraticProgram
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer
#IBMQ.load_account() # Load account from disk
#IBMQ.providers()    # List all available providers
from qiskit.optimization.applications.ising.docplex import *
from qiskit.aqua.components.initial_states import Custom
from grid_search import *

# IBMQ.stored_account()

pauli_z = [[1, 0], [0, -1]]
print(pauli_z)

pauli_x = [[0, 1], [1, 0]]
pauli_y1 =[]




import time
global n

# Another way to include
n     = 3
V     = np.arange(0,n,1)
E     =[(0,1,1.0),(0,2,1.0),(1,2,1.0)]

global G 
G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)

E = G.edges()


# Generate plot of the Graph
colors       = ['r' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos          = nx.spring_layout(G)

nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

global n_layers, length
n_layers = 1
print("\np={:d}".format(n_layers))

global adj
adj = read_graph(n)


global qubitOp

mdl = Model(name='tsp-1')
x2 = {i: mdl.binary_var() for i in range(n**2) } #the possible bitstrings

# Object function
tsp_func = mdl.sum(adj[i][j] * x2[t*n +i]*x2[(t+1)*n + j]  for i in range(n) for j in range(n) for t in range(n-1))
mdl.minimize(tsp_func)
        
# Constraints

for t in range(n):
    mdl.add_constraint(mdl.sum(x2[t*n+i] for i in range(n)) == 1)
for i in range(n):
    mdl.add_constraint(mdl.sum(x2[t*n+i] for t in range(n)) == 1)
    
qubitOp, offset = get_operator(mdl)     
print('Offset:', offset)
print('Ising Hamiltonian:')
print(qubitOp.print_details())



sample = generate_grid_2(1,0.1)
f = [objective(par) for par in sample]




best_index = 0

for i in range(len(sample)):
    if f[i] < f[best_index]:
        best_index = i
        
print(sample[best_index])


# creates circuit using qaoa instance and optimal parameters
aqua_globals.random_seed = 10598
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   seed_simulator=aqua_globals.random_seed,
                                   seed_transpiler=aqua_globals.random_seed)
qaoa_mes = QAOA(quantum_instance=quantum_instance,operator = qubitOp,p=n_layers)  #initial_state = initial #initial_point=[0.,0.]
circuit2 = QuantumCircuit(n**2)
circuit = qaoa_mes.construct_circuit(sample[best_index])
    
reg = ClassicalRegister(n**2)
circuit[0].add_register(reg)
circuit[0].measure(range(n**2),range(n**2))


# Execution of circuit(either simulation or on real quantum machines)
backend      = Aer.get_backend("qasm_simulator")
shots        = 1024
simulate     = execute(circuit[0], backend=backend, shots=shots)
QAOA_results = simulate.result()


counts = QAOA_results.get_counts()
counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}  #sort the counts dictionary


#finds top 10 probable bitstrings and print them
clist = []
for i in counts.items():
    clist.append(i)
counts = {}  
for i in range(len(clist)-1,len(clist)-11,-1):
    print(clist[i])
    
    
#print((time2-time1))







