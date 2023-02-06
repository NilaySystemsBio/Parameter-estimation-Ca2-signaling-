# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:33:04 2018
@author: Nilay Kumar
Designation: Graduate Student
Department: Chemical and Biomolecular Engineering
School: University of Notre Dame
"""

"""
CODE DESCRIPTION:
	INPUT: 
	1. The code takes in input as excel sheet cpntaining the level of cytosolic calcium with time. 
	2. The first row of the excel sheet contains the time (in minutes) and Ca2+ data from experiments
	3. The experimental data has to be normalized so as to match the bounds of the physical model whose 
	parameter has to be calibrated. One poolmodelwas used (see modeldescription.pdf)
	
	The code uses Pyomo (http://www.pyomo.org/) for finding the parameters of the physical Ca2+ model that best represent the experimental data
	
	OUTPUT:
		The following plots are generated as an output.
			1. 
"""

"""
STEP 0: Importing python libraries and Pyomo
"""
# Importing pandas
import pandas as pd
# Importing pyomo libraries
from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator
# Importing matplotlib for plotting
import matplotlib.pyplot as plt

"""
STEP 1: User inputs
"""
# The file that contains the experimental data
filename_Ca_data = 'input_data_files/data_n.xlsx'
# Initial values of the model variables
initial_value_y = 1
initial_value_z = 0.6 # Ca2+ (measured variable in experiment)

"""
STEP 2: Defining functions for task of optimization
"""

'''
 Declaring model parameters and variables 
(see modeldescription.pdf)
'''
class OnepoolResults:    
    # Model parameters
    beta = 0.0 # a.u
    mu_0 = 0.0 # micro-M/min
    mu_1 = 0.0 # micro-M/min
    vm_2 = 0.0 # micro-M/min
    vm_3 = 0.0 # micro-M/min
    k_2 = 0.0 # micro-M
    k_r = 0.0 # micro-M
    k_a = 0.0 # micro-M
    k = 0.0 # 1/min
    k_f = 0.0 # 1/min
    param_n = 0.0 # a.u.
    param_m = 0.0 # a.u.
    param_p = 0.0 # a.u.
    
    # Model variables/optimization results
    time = [] # in min
    Z = [] # micro-M
    Y = [] # micro-M
    v2 = []
    v3 = []

"""
Function is used to load the data from the excel sheet
Fisrt column: of the excel sheet should have heading time and should contain time inminutes
Second column: Contains Normalized Ca2+ data. The Ca2+ data has to be normalized to be in bounds of the model outputs.
               We leave it to users on the choice of normalization strategies
The sheet containing data should benamed calcium

Output: time and Ca rrays containing the experimental data			            
"""
def load_data():	
    # Time step
    df_items1 = pd.read_excel(filename_Ca_data, sheet_name='calcium', header=0, index_col=0)    
    # Ca concentration, [scaled units]
    df_items2 = pd.read_excel(filename_Ca_data, sheet_name='calcium', header=0, index_col=1)
    time_data = df_items1.index.tolist()
    ca_data = df_items2.index.tolist()
    # onepool = dict(zip(time_data,ca_data))
    return time_data, ca_data

"""
Input: 
	tf -> Total time of simulation (time of experimental data)
	
Tasks:
	Defines a model within pyome:
		Defines parameters, variables and equations of the physical model

Output:
	The physical model
"""
def create_model(tf):
    m = ConcreteModel() # defining the mdel in pyomo    	      
    # Defining the timr whwr the model will be simulated as a continuosu variablei in pyomo
    m.time = ContinuousSet(bounds=(0,tf))    
    m.onepoolmeas = Param(m.time, initialize=0.0, mutable=True)    
    # Cytosolic Ca concentration, [micro-M]
    m.Z = Var(m.time, bounds=(0.01, 5))    
    # ER Calcium concentration
    m.Y = Var(m.time, bounds=(0.1, 10))    
    # Fitted parameter and their bounds and initialization values
    m.beta = Var(within=NonNegativeReals, bounds = (0.1,1), initialize = 0.4) # a.u.  
    m.mu_0 = Var(within=NonNegativeReals, bounds = (0.1,10), initialize = 3.4) # micro-M/min
    m.mu_1 = Var(within=NonNegativeReals, bounds = (0.1,10), initialize = 3.4) # micro-M/min
    m.vm_2 = Var(within=NonNegativeReals, bounds = (0.1,100), initialize = 50) # micro-M/min
    m.vm_3 = Var(within=NonNegativeReals, bounds = (0.1,1000), initialize = 650) # micro-M/min
    m.k_2 = Var(within=NonNegativeReals, bounds = (0.1,10), initialize = 1) # micro-M
    m.k_r = Var(within=NonNegativeReals, bounds = (0.1,10), initialize = 2) # micro-M
    m.k_a = Var(within=NonNegativeReals, bounds = (0.1,1), initialize = 0.9) # micro-M
    m.k = Var(within=NonNegativeReals, bounds = (0.1,100), initialize = 10) # 1/min
    m.k_f = Var(within=NonNegativeReals, bounds = (0.1,10), initialize = 1) # 1/min
    m.param_n = Var(within=NonNegativeReals, bounds = (0.1,5), initialize = 2) #a.u.
    m.param_m = Var(within=NonNegativeReals, bounds = (0.1,5), initialize = 2) # a.u.
    m.param_p = Var(within=NonNegativeReals, bounds = (0.1,10), initialize = 4) # a.u.
    
    # Initializing the derivatives of variables with time
    m.Ydot = DerivativeVar(m.Y,wrt=m.time) # dERCa/dt
    m.Zdot = DerivativeVar(m.Z,wrt=m.time) # dCa/dt   
    m.v2 = Var(m.time) #
    m.v3 = Var(m.time) #    
    
    # Defining the initial conditions
    def _init_conditions(m):
        yield m.Y[0] == 1 # Initial ER Calcium concetration
        yield m.Z[0] == 0.6 # Initial calcium concentration
    m.init_conditions = ConstraintList(rule = _init_conditions)
    
    # Differential equations qithin the modell (see model description.pdf)
    def _Ydot(m,i):
    #    v2 = ((m.vm_2)*(((m.Z[i])**(m.param_n))/(((m.k_2)**(m.param_n))+((m.Z[i])**(m.param_n)))));
    #   v3 = ((((m.Y[i])**(m.param_m))/(((m.k_r)**(m.param_m))+((m.Y[i])**(m.param_m))))*(((m.Z[i])**(m.param_p))/(((m.k_a)**(m.param_p))+((m.Z[i])**(m.param_p))))*(m.vm_3))
        return m.Ydot[i] == m.v2[i] - m.v3[i] - ((m.k_f)*(m.Y[i]))
    m.Ydotcon = Constraint(m.time, rule = _Ydot)
        
    def _Zdot(m,i):
        vin = m.mu_0 + ((m.mu_1)*(m.beta))
    #    v2 = ((m.vm_2)*(((m.Z[i])**(m.param_n))/(((m.k_2)**(m.param_n))+((m.Z[i])**(m.param_n)))))
    #    v3 = ((((m.Y[i])**(m.param_m))/(((m.k_r)**(m.param_m))+((m.Y[i])**(m.param_m))))*(((m.Z[i])**(m.param_p))/(((m.k_a)**(m.param_p))+((m.Z[i])**(m.param_p))))*(m.vm_3)) 
        return m.Zdot[i] ==  vin - m.v2[i]  + m.v3[i] + ((m.k_f)*(m.Y[i])) -((m.k)*(m.Z[i]))
    m.Zdotcon = Constraint(m.time, rule = _Zdot)
    
    # Algebraic equations
    def _v2(m, i):
        return (((m.k_2)**(m.param_n))+((m.Z[i])**(m.param_n)))*m.v2[i] == (m.vm_2)*(((m.Z[i])**(m.param_n)))
    m.v2def = Constraint(m.time, rule=_v2)
    
    def _v3(m, i):
        return (((m.k_r)**(m.param_m))+((m.Y[i])**(m.param_m)))*(((m.k_a)**(m.param_p))+((m.Z[i])**(m.param_p)))*m.v3[i] == ((m.Y[i])**(m.param_m))*((m.Z[i])**(m.param_p))*(m.vm_3)
    m.v3def = Constraint(m.time, rule=_v3)
    # Returns the model
    return m


"""
Input: The physical model 
The model used Pyomo simulator to simulate the physical model at the defined number of poits

"""
def simulate_model(m):    
    ## Need initial guess for parameters    
    m.var_input = Suffix(direction=Suffix.LOCAL)
    m.Y[0] = initial_value_y
    m.Z[0] = initial_value_z # Initial value of calcium
    sim = Simulator(m, package='casadi') # simulate themodel
    tsim, profiles = sim.simulate(numpoints=1000, integrator='idas') # generate time and model outputs 
    # Return the simulates profles generate dusing model
    return sim, tsim, profiles


"""
INPUTs:
	A) m: The physical model for Ca2+ signaling
	B) time_data: The time points at which Ca2+ was measured (min))
	C) Ca2+ data: The measured experimental ca2+ data at the time points
	
TASK: The portion of code uses pyomo to estimate the best model parameters that generate the 
		experimental data using a least square objective function.
		
OUTPUTS:
	A) m: The best model corresponding the experimental data
	B) results: The fitted parameter values
"""
def fit_parameters(m,time_data,ca_data,sim):

    ### Discretize and optimize
    discretizer = TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m,nfe=96,scheme='BACKWARD')
    sim.initialize_model()

    if isinstance(sim,pyomo.dae.simulator.Simulator):
        sim.initialize_model()
        
    # Now that time is discretized, copy data into m.onepoolmeas
    for i,t in enumerate(m.time):    
        m.onepoolmeas[t] = ca_data[i]

    # Need to move objective here. It should be written in discretized time.
    def _obj(m):
        return sum((m.Z[i] - m.onepoolmeas[i])**2  for i in m.time if i != 0)
    m.obj = Objective(rule=_obj)
    
    # Use default ipopt
    solver=SolverFactory('ipopt')
	
    """
    # Specify location of ipopt executable - Nilay comment this out
    #solver=SolverFactory('ipopt', executable="/Users/adowling/src/CoinIpopt/bin/ipopt")    
    # Use HSL MA57 linear algebra - comment this out
    #solver.options['linear_solver'] = "ma57"
    """
    # Set maximum number of iterations to 10000
    solver.options['max_iter'] = 10000    
    # Solve model
    results = solver.solve(m, tee=True)
    # Return the optimized model and fitted parameters as results
    return m, results


"""
INPUTS: 
	m: The best model that fits the experimental data
	
OUTPUTS:
	A) time_model: The model is evaluated over the times at which the experimental data was calculated
	B) Z_data:; The model is used to calculate Ca2+ at those points
"""
def extract_results(m):
    # Accessing the solution (model predictions for Ca and time)
    time_model = [time for time in m.time]
    Z_model = [m.Z[time]() for time in m.time]
    Z_data = [m.onepoolmeas[time]() for time in m.time]
    # Returns time and Ca2+ values predicted by the model to fit the experimental data
    return time_model, Z_model, Z_data

"""
The function plots the experimental data along side the best fit model results 
"""
def plot_results(tpredict,Zpredict,tdata,Zdata,plot_title):

    plt.figure() # defining the figure using matplotlib
    plt.plot(tpredict, Zpredict, label = 'Model predictions') # plotting the model predictions
    plt.scatter(tdata, Zdata, label='Experimental data', c='red') # Plotting the experimental data
    plt.xlabel('time (in min)', fontsize = 15) # Labeling x axis
    plt.ylabel('$Ca^{2+}$ $(\mu M)$', fontsize = 15) # Labeling y axis
    plt.title(plot_title, fontsize = 15) # Creating the title of plot
    plt.legend() # Show legends
    plt.show() 


"""
Solving the physical modelnumerically yusing an explicit finite difference scheme
INPUTS:
A) tfinalsec: Input the total time of simulation in seconds
B) beta, mu_0, mu_1, vm_2, vm_3, k_2, k_r, k_a, k, k_f, n, m, p: The inputs to this
	 finction are the parameters of the phusical model defined in model_description.pdf

OUTPUTS: Returns the time and Ca2+ concentration

"""

def single_pool(tfinalsec,beta, mu_0, mu_1, vm_2, vm_3, k_2, k_r, k_a, k, k_f, n, m, p):            
    
    # calling libraries
    import numpy as np
    import matplotlib.pyplot as plt
    
    #defining temporal gid resolution
    t_final = tfinalsec/60
    delta_t = 0.001
    n_steps = int(t_final/delta_t)

    # Initializing arrays for storing Ca2+ concentartions
    z = np.zeros ( (n_steps,1) )  # Cytosolic
    y = np.zeros ( (n_steps,1) )  # Endoplasmic reticulum
    time = np.zeros ( (n_steps,1) )

    # Initial conditions. Can be cosidered as parameters too/
    z[0] = 0.6
    y[0] = 1

    # Explicit 1st order finite difference scheme 
    for i in range(1, n_steps):
        v_2 = (vm_2*((z[i-1])**n))/(k_2**n + ((z[i-1])**n))
        v_3 = ((vm_3*((z[i-1])**p))/(k_a**p + ((z[i-1])**p)))*((((y[i-1])**m))/(k_r**m + ((y[i-1])**m)))
        v_in = mu_0 + mu_1*beta
        z[i] = z[i-1] + delta_t*(v_in - v_2 + v_3 + k_f*y[i-1] - k*z[i-1])
        y[i] = y[i-1] + delta_t*(v_2 - v_3 - k_f*y[i-1])
        time[i] = time[i-1] + delta_t
    return z, y, time


"""
STEP 2: Putting it all together
	A) Load the data
	B) Create the physical model
	C) Generate some initial profiels using the modle to identify thebest initial conditions
	D) Fit the data using the pyomo based framework
	E) Plot the results
	G) Print parameter values
"""
if __name__ == "__main__":
    
    # Load data
    time_data, ca_data = load_data()    
    # Create Pyomo model
    m = create_model(1.59)    
    # Simulating the model using pyomos solvers for finding thebest initial condition
    sim, tsim, profiles = simulate_model(m)                
    # Discretize model, optimize to fit parameters
    m, results = fit_parameters(m,time_data,ca_data, sim)    
    # Extract solution from model
    time_model, Z_model, Z_data = extract_results(m)    
    # Plot results
    plot_results(time_model,Z_model,time_model,Z_data,"Estimated model predictions")    
    # Call manual simulation routine
    z_mod, y_mod, time_mod = single_pool(96, value(m.beta), value(m.mu_0), value(m.mu_1), value(m.vm_2), value(m.vm_3), value(m.k_2), value(m.k_r), value(m.k_a), value(m.k), value(m.k_f), value(m.param_n), value(m.param_m), value(m.param_p))