#!pip install pyomo
#!pip install cplex

from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
from pyomo import environ as pe
import os
import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")
import seaborn as sns
import datetime
import gurobipy

dff = pd.read_excel("C:/Users/Baturalp/Desktop/oildata.xlsx")

df = pd.DataFrame()

df[["x1", "x2", "x3"]] = dff[["EXCHANGE RATE", "INFLATION RATE (%)", "BRENT CRUDE OIL PRICE"]] #1., 2. AND 3. INDEPENDENT VARIABLES
df["y"] = dff["OIL PRICE"]

df = df[24:66]
df.index = np.arange(0,42)

df["x1"] = df["x1"].astype("float64")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() #SCALING INDEPENDENT VARIABLES
df["x1"] = scaler.fit_transform(df[["x1"]])
df["x2"] = scaler.fit_transform(df[["x2"]])
df["x3"] = scaler.fit_transform(df[["x3"]])

df_test = df[33:]
df_test.index = np.arange(0,9)

df = df[:33]
N = df.count()[0]
dfcos = df.copy()

plt.figure(figsize=(16,5))

#FIRST INDEPENDENT VARIABLE - DEPENDENT VARIABLE GRAPH
plt.subplot(1,3,1)
plt.scatter(df["x1"], df['y'], s= 100, color = "orange")
plt.xlabel('  EXCHANGE RATE ', fontsize=12, fontweight='bold')
plt.ylabel('  OIL PRICE ', fontsize=12, fontweight='bold')
plt.xticks( fontsize=12, fontweight='bold')
plt.yticks( fontsize=12, fontweight='bold')
plt.grid()
plt.tight_layout()

#SECOND INDEPENDENT VARIABLE - DEPENDENT VARIABLE GRAPH
plt.subplot(1,3,2)
plt.scatter(df["x2"], df['y'], s= 100, color = "green")
plt.xlabel('  INFLATION RATE ', fontsize=12, fontweight='bold')
plt.ylabel('  OIL PRICE ', fontsize=12, fontweight='bold')
plt.xticks( fontsize=12, fontweight='bold')
plt.yticks( fontsize=12, fontweight='bold')
plt.grid()
plt.tight_layout()

#THIRD INDEPENDENT VARIABLE - DEPENDENT VARIABLE GRAPH
plt.subplot(1,3,3)
plt.scatter(df["x3"], df['y'], s= 100, color = "red")
plt.xlabel('  BRENT CRUDE OIL PRICE ', fontsize=12, fontweight='bold')
plt.ylabel('  OIL PRICE ', fontsize=12, fontweight='bold')
plt.xticks( fontsize=12, fontweight='bold')
plt.yticks( fontsize=12, fontweight='bold')
plt.grid()
plt.tight_layout()

plt.savefig('Dots '+  '.jpg', format='jpg', dpi=400)
plt.show()

#BUILDING THE MODEL
#In this part, we will build our multi-curve regression model. 
#First, since we have three independent variables, 
#we set up the model accordingly. We use gurobi as the solver and 
#adjust the number of curves with K and the polynomial degree with M. 
#We set the timelimit for how long the model will run. 
#(This is an example, so the model was run very briefly.)

rs = 42
BM= 10000 #BIG M
K= 3 #K NUMBER OF CURVES
M = 1 #M DEGREE OF THE POLYNOMIAL
plt.figure(figsize=(8,5))
counter =0 
for M in [M]:
    for K in [K]:
        counter +=1
        model = AbstractModel()
        model.i = RangeSet(N)
        model.m = RangeSet(M+1)
        model.k = RangeSet(K)
        model.target = Param(model.i, initialize=0, mutable=True)
        model.r = Var(model.i,bounds=(0, 1000) , within = Reals)
        
        model.c1 = Var(model.k, model.m, bounds=(-100,100) , within = Reals) #COEFFICIENT 1 (C1) DEFINITION
        model.c2 = Var(model.k, model.m, bounds=(-100,100) , within = Reals) #COEFFICIENT 2 (C2) DEFINITION
        model.c3 = Var(model.k, model.m, bounds=(-100,100) , within = Reals) #COEFFICIENT 3 (C3) DEFINITION
        
        model.U = Var(model.i, model.k,bounds=(0, 1) , within = Binary)
        def rule_C1(model,i,k):
            return (sum(model.c1[k,m]*df.loc[i-1,'x1']**(m-1) for m in model.m ) + #1. CONSTRAINT 1. INDEPENDENT VARIABLE
                   sum(model.c2[k,m]*df.loc[i-1,'x2']**(m-1) for m in model.m ) +  #1. CONSTRAINT 2. INDEPENDENT VARIABLE
                   sum(model.c3[k,m]*df.loc[i-1,'x3']**(m-1) for m in model.m )    #1. CONSTRAINT 3. INDEPENDENT VARIABLE
                   - df.loc[i-1,'y'] <= model.r[i] + BM*(1-model.U[i,k]))
        model.C1 = Constraint(model.i, model.k, rule=rule_C1)
        def rule_C2(model,i,k):
            return (sum(model.c1[k,m]*df.loc[i-1,'x1']**(m-1) for m in model.m ) + #2. CONSTRAINT 1. INDEPENDENT VARIABLE
                   sum(model.c2[k,m]*df.loc[i-1,'x2']**(m-1) for m in model.m ) +  #2. CONSTRAINT 2. INDEPENDENT VARIABLE
                   sum(model.c3[k,m]*df.loc[i-1,'x3']**(m-1) for m in model.m )    #2. CONSTRAINT 3. INDEPENDENT VARIABLE
                   - df.loc[i-1,'y'] >= -model.r[i] - BM*(1-model.U[i,k]))
        model.C2 = Constraint(model.i,model.k,  rule=rule_C2)
        def rule_C3(model,i):
            return sum(model.U[i,k] for k in model.k) == 1
        model.C3 = Constraint(model.i,  rule=rule_C3)
        def rule_C4(model,k):
            if k>1:
                return sum(model.U[i,k-1] for i in model.i) <= sum(model.U[i,k] for i in model.i)
            else:
                return Constraint.Skip
        model.C4 = Constraint(model.k,  rule=rule_C4)
        
        def rule_C5(model,k):
            return sum(model.U[i,k] for i in model.i) >= 1
        model.C5 = Constraint(model.k,  rule=rule_C5)
        
        def rule_of(model):
            return sum(model.r[i] for i in model.i)
        model.obj = Objective(rule=rule_of, sense=minimize)

        instance = model.create_instance()
        gurobi = pe.SolverFactory('gurobi', solver_io = "python")
        gurobi.options['mipgap'] = 0.05
        gurobi.options["TimeLimit"] = 0.2*60
        results = gurobi.solve(instance, tee=True, keepfiles=True)  
                
        print('OF= ',value(instance.obj)) #OBJECTIVE FUNCTION
        
print("----------------------------------------------------------------------------")

#Then, we need to create a dataframe that shows us
#independent variables, dependent variable, prediction, residue and index's assigned line.
#For prediction, residue and line, we create an empty array then fill them with the equations.
#And with codes below there, we get polynomial equations for models

dfy = pd.DataFrame()
dfy["E. Rate"] = df["x1"] #EXCHANGE RATE
dfy["I. Rate"] = df["x2"] #INFLATION RATE
dfy["B. Price"] = df["x3"] #BRENT CRUDE OIL PRICE
dfy["Price"] = df["y"] #OIL PRICE
dfy["Prediction"] = np.arange(0,N)
dfy["Residue"] = np.arange(0,N)
dfy["Line"] = np.arange(0,N)

if K == 1:
    if M == 1:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            dfy["Line"][i] = "First Line"
            dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + 
                                    value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] +
                                    value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i])
            
        print("Polynomial Equation = ",round(value(instance.c1[1,2]),2),"x1 +", round(value(instance.c2[1,2]),2),"x2 +", 
              round(value(instance.c3[1,2]),2),"x3 +", round(value(instance.c1[1,1]),2), "+", 
              round(value(instance.c2[1,1]),2),"+", round(value(instance.c3[1,1]),2))
    
            
    elif M == 2:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            dfy["Line"][i] = "First Line"
            dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + value(instance.c1[1,3]*df["x1"][i]**2) +
                                    value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] + value(instance.c2[1,3]*df["x2"][i]**2) +
                                    value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i] + value(instance.c3[1,3]*df["x3"][i]**2))
        print("Polynomial Equation = ",round(value(instance.c1[1,3]),2), "x1^2 +", round(value(instance.c2[1,3]),2), "x2^2 +",
              round(value(instance.c3[1,3]),2), "x3^2 +", round(value(instance.c1[1,2]),2),"x1 +", 
              round(value(instance.c2[1,2]),2),"x2 +", round(value(instance.c3[1,2]),2),"x3 +", 
              round(value(instance.c1[1,1]),2), "+", round(value(instance.c2[1,1]),2),"+", round(value(instance.c3[1,1]),2))
            
            
    elif M == 3:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            dfy["Line"][i] = "First Line"
            dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + value(instance.c1[1,3]*df["x1"][i]**2) +
                                    value(instance.c1[1,4])*df["x1"][i]**3 +
                         
                                    value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] + value(instance.c2[1,3]*df["x2"][i]**2) +
                                    value(instance.c2[1,4])*df["x2"][i]**3 +
                         
                                    value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i] + value(instance.c3[1,3]*df["x3"][i]**2) +
                                    value(instance.c3[1,4])*df["x3"][i]**3)
        print("Polynomial Equation = ",round(value(instance.c1[1,4]),2), "x1^3 +", round(value(instance.c2[1,4]),2), "x2^3 +",
              round(value(instance.c3[1,4]),2), "x3^3 +",round(value(instance.c1[1,3]),2), "x1^2 +", round(value(instance.c2[1,3]),2), "x2^2 +",
              round(value(instance.c3[1,3]),2), "x3^2 +",round(value(instance.c1[1,2]),2),"x1 +",round(value(instance.c2[1,2]),2),"x2 +", round(value(instance.c3[1,2]),2),"x3 +", 
              round(value(instance.c1[1,1]),2), "+", round(value(instance.c2[1,1]),2),"+", round(value(instance.c3[1,1]),2))
            
            
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################         
            
elif K == 2:
    if M == 1:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            if value(instance.U[i+1,1]) == 1:
                dfy["Line"][i] = "First Line"
                dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + 
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i])
            elif value(instance.U[i+1,2]) == 1:
                dfy["Line"][i] = "Second Line"
                dfy["Prediction"][i] = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df["x1"][i] + 
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df["x2"][i] +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df["x3"][i])
        for i in np.arange(1,K+1):
            print("Polynomial Equation = ",round(value(instance.c1[i,2]),2),"x1 +", round(value(instance.c2[i,2]),2),"x2 +", 
            round(value(instance.c3[i,2]),2),"x3 +", round(value(instance.c1[i,1]),2), "+", 
            round(value(instance.c2[i,1]),2),"+", round(value(instance.c3[i,1]),2))
        
    
    
    elif M == 2:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            if value(instance.U[i+1,1]) == 1:
                dfy["Line"][i] = "First Line"
                dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + value(instance.c1[1,3]*df["x1"][i]**2) +
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] + value(instance.c2[1,3]*df["x2"][i]**2) +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i] + value(instance.c3[1,3]*df["x3"][i]**2))
            elif value(instance.U[i+1,2]) == 1:
                dfy["Line"][i] = "Second Line"
                dfy["Prediction"][i] = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df["x1"][i] + value(instance.c1[2,3]*df["x1"][i]**2) +
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df["x2"][i] + value(instance.c2[2,3]*df["x2"][i]**2) +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df["x3"][i] + value(instance.c3[2,3]*df["x3"][i]**2))
        for i in np.arange(1,K+1):
            print("Polynomial Equation = ",round(value(instance.c1[i,3]),2), "x1^2 +", round(value(instance.c2[i,3]),2), "x2^2 +",
            round(value(instance.c3[i,3]),2), "x3^2 +", round(value(instance.c1[i,2]),2),"x1 +", 
            round(value(instance.c2[i,2]),2),"x2 +", round(value(instance.c3[i,2]),2),"x3 +", 
            round(value(instance.c1[i,1]),2), "+", round(value(instance.c2[i,1]),2),"+", round(value(instance.c3[i,1]),2))
        
        
    elif M == 3:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            if value(instance.U[i+1,1]) == 1:
                dfy["Line"][i] = "First Line"
                dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + value(instance.c1[1,3]*df["x1"][i]**2) +
                                        value(instance.c1[1,4])*df["x1"][i]**3 +
                         
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] + value(instance.c2[1,3]*df["x2"][i]**2) +
                                        value(instance.c2[1,4])*df["x2"][i]**3 +
                         
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i] + value(instance.c3[1,3]*df["x3"][i]**2) +
                                        value(instance.c3[1,4])*df["x3"][i]**3)
            elif value(instance.U[i+1,2]) == 1:
                dfy["Line"][i] = "Second Line"
                dfy["Prediction"][i] = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df["x1"][i] + value(instance.c1[2,3]*df["x1"][i]**2) +
                                        value(instance.c1[2,4])*df["x1"][i]**3 +
                         
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df["x2"][i] + value(instance.c2[2,3]*df["x2"][i]**2) +
                                        value(instance.c2[2,4])*df["x2"][i]**3 +
                         
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df["x3"][i] + value(instance.c3[2,3]*df["x3"][i]**2) +
                                        value(instance.c3[2,4])*df["x3"][i]**3)
        
        for i in np.arange(1,K+1):
            print("Polynomial Equation = ",round(value(instance.c1[i,4]),2), "x1^3 +", round(value(instance.c2[i,4]),2), "x2^3 +",
                  round(value(instance.c3[i,4]),2), "x3^3 +",round(value(instance.c1[i,3]),2), "x1^2 +", round(value(instance.c2[i,3]),2), "x2^2 +",
                  round(value(instance.c3[i,3]),2), "x3^2 +",round(value(instance.c1[i,2]),2),"x1 +",round(value(instance.c2[i,2]),2),"x2 +", round(value(instance.c3[i,2]),2),"x3 +", 
                  round(value(instance.c1[i,1]),2), "+", round(value(instance.c2[i,1]),2),"+", round(value(instance.c3[i,1]),2))
        
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


elif K == 3:
    if M == 1:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            if value(instance.U[i+1,1]) == 1:
                dfy["Line"][i] = "First Line"
                dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + 
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i])
            elif value(instance.U[i+1,2]) == 1:
                dfy["Line"][i] = "Second Line"
                dfy["Prediction"][i] = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df["x1"][i] + 
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df["x2"][i] +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df["x3"][i])
            elif value(instance.U[i+1,3]) == 1:
                dfy["Line"][i] = "Third Line"
                dfy["Prediction"][i] = (value(instance.c1[3,1]) + value(instance.c1[3,2])*df["x1"][i] + 
                                        value(instance.c2[3,1]) + value(instance.c2[3,2])*df["x2"][i] +
                                        value(instance.c3[3,1]) + value(instance.c3[3,2])*df["x3"][i])
        for i in np.arange(1,K+1):
            print("Polynomial Equation = ",round(value(instance.c1[i,2]),2),"x1 +", round(value(instance.c2[i,2]),2),"x2 +", 
                  round(value(instance.c3[i,2]),2),"x3 +", round(value(instance.c1[i,1]),2), "+", 
                  round(value(instance.c2[i,1]),2),"+", round(value(instance.c3[i,1]),2))
                
                
    elif M == 2:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            if value(instance.U[i+1,1]) == 1:
                dfy["Line"][i] = "First Line"
                dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + value(instance.c1[1,3]*df["x1"][i]**2) +
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] + value(instance.c2[1,3]*df["x2"][i]**2) +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i] + value(instance.c3[1,3]*df["x3"][i]**2))
            elif value(instance.U[i+1,2]) == 1:
                dfy["Line"][i] = "Second Line"
                dfy["Prediction"][i] = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df["x1"][i] + value(instance.c1[2,3]*df["x1"][i]**2) +
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df["x2"][i] + value(instance.c2[2,3]*df["x2"][i]**2) +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df["x3"][i] + value(instance.c3[2,3]*df["x3"][i]**2))
            elif value(instance.U[i+1,3]) == 1:
                dfy["Line"][i] = "Third Line"
                dfy["Prediction"][i] = (value(instance.c1[3,1]) + value(instance.c1[3,2])*df["x1"][i] + value(instance.c1[3,3]*df["x1"][i]**2) +
                                        value(instance.c2[3,1]) + value(instance.c2[3,2])*df["x2"][i] + value(instance.c2[3,3]*df["x2"][i]**2) +
                                        value(instance.c3[3,1]) + value(instance.c3[3,2])*df["x3"][i] + value(instance.c3[3,3]*df["x3"][i]**2))
        for i in np.arange(1,K+1):
            print("Polynomial Equation = ",round(value(instance.c1[i,3]),2), "x1^2 +", round(value(instance.c2[i,3]),2), "x2^2 +",
            round(value(instance.c3[i,3]),2), "x3^2 +", round(value(instance.c1[i,2]),2),"x1 +", 
            round(value(instance.c2[i,2]),2),"x2 +", round(value(instance.c3[i,2]),2),"x3 +", 
            round(value(instance.c1[i,1]),2), "+", round(value(instance.c2[i,1]),2),"+", round(value(instance.c3[i,1]),2))
                
                
    elif M == 3:
        for i in np.arange(0,N):
            dfy["Residue"][i] = value(instance.r[i+1])
            if value(instance.U[i+1,1]) == 1:
                dfy["Line"][i] = "First Line"
                dfy["Prediction"][i] = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df["x1"][i] + value(instance.c1[1,3]*df["x1"][i]**2) +
                                        value(instance.c1[1,4])*df["x1"][i]**3 +
                         
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df["x2"][i] + value(instance.c2[1,3]*df["x2"][i]**2) +
                                        value(instance.c2[1,4])*df["x2"][i]**3 +
                         
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df["x3"][i] + value(instance.c3[1,3]*df["x3"][i]**2) +
                                        value(instance.c3[1,4])*df["x3"][i]**3)
            elif value(instance.U[i+1,2]) == 1:
                dfy["Line"][i] = "Second Line"
                dfy["Prediction"][i] = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df["x1"][i] + value(instance.c1[2,3]*df["x1"][i]**2) +
                                        value(instance.c1[2,4])*df["x1"][i]**3 +
                         
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df["x2"][i] + value(instance.c2[2,3]*df["x2"][i]**2) +
                                        value(instance.c2[2,4])*df["x2"][i]**3 +
                         
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df["x3"][i] + value(instance.c3[2,3]*df["x3"][i]**2) +
                                        value(instance.c3[2,4])*df["x3"][i]**3)
            elif value(instance.U[i+1,3]) == 1:
                dfy["Line"][i] = "Third Line"
                dfy["Prediction"][i] = (value(instance.c1[3,1]) + value(instance.c1[3,2])*df["x1"][i] + value(instance.c1[3,3]*df["x1"][i]**2) +
                                        value(instance.c1[3,4])*df["x1"][i]**3 +
                         
                                        value(instance.c2[3,1]) + value(instance.c2[3,2])*df["x2"][i] + value(instance.c2[3,3]*df["x2"][i]**2) +
                                        value(instance.c2[3,4])*df["x2"][i]**3 +
                         
                                        value(instance.c3[3,1]) + value(instance.c3[3,2])*df["x3"][i] + value(instance.c3[3,3]*df["x3"][i]**2) +
                                        value(instance.c3[3,4])*df["x3"][i]**3)
        for i in np.arange(1,K+1):
            print("Polynomial Equation = ",round(value(instance.c1[i,4]),2), "x1^3 +", round(value(instance.c2[i,4]),2), "x2^3 +",
                  round(value(instance.c3[i,4]),2), "x3^3 +",round(value(instance.c1[i,3]),2), "x1^2 +", round(value(instance.c2[i,3]),2), "x2^2 +",
                  round(value(instance.c3[i,3]),2), "x3^2 +",round(value(instance.c1[i,2]),2),"x1 +",round(value(instance.c2[i,2]),2),"x2 +", round(value(instance.c3[i,2]),2),"x3 +", 
                  round(value(instance.c1[i,1]),2), "+", round(value(instance.c2[i,1]),2),"+", round(value(instance.c3[i,1]),2))
        
print("----------------------------------------------------------------------------")
print(dfy)
dfy.to_excel(R"C:\Users\Baturalp\Desktop\df.xlsx")

#After all that calculations and codes, we create a graph that shows us
#model's predictions and real oil prices according to date.

dfy.index = np.arange('2020-01', '2022-10', dtype='datetime64[M]')

fl = dfy[dfy["Line"] == "First Line"]

if K == 2:
    sl = dfy[dfy["Line"] == "Second Line"]
elif K == 3:
    sl = dfy[dfy["Line"] == "Second Line"]
    tl = dfy[dfy["Line"] == "Third Line"]

x = np.arange('2020-01', '2022-10', dtype='datetime64[M]')

plt.figure(figsize = (8,5))
plt.plot(x, dfy["Price"], color = "black")
plt.plot(x, dfy["Prediction"], color = "black")
plt.scatter(dfy["Price"].index, dfy["Price"], color = "darkblue", s = 75, zorder = 2)
plt.scatter(fl["Prediction"].index, fl["Prediction"], color = "orange", s = 75, zorder = 2)
plt.scatter(sl["Prediction"].index, sl["Prediction"], color = "green", s = 75, zorder = 2)
plt.scatter(tl["Prediction"].index, tl["Prediction"], color = "red", s = 75, zorder = 2)
plt.xticks( fontsize=12, fontweight='bold')
plt.yticks( fontsize=12, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.xlabel("DATE", fontsize=12, fontweight='bold')
plt.ylabel("OIL PRICE", fontsize=12, fontweight='bold')
plt.savefig('k1m1 '+  '.jpg', format='jpg', dpi=300)
plt.show()

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
print(df)

#COSINE SIMILARITY
#In this section, we will use cosine similarity for predicting oil prices.
#First, a cosine similarity study is performed between each test variable and 
#all training variables, and the test variable is assigned to the curve that 
#gives the highest cosine similarity result for the test variable.
#Right there, we will split the dataset for %35 test, %65 train
#then see how many true results can this model give us.

dfcos["Line"] = np.arange(0,N)
for i in np.arange(0,N):
    dfcos["Line"][i] = dfy["Line"][i]

from sklearn.model_selection import train_test_split

x_train_pred, x_test_pred, y_train_pred, y_test_pred = train_test_split(dfy[["E. Rate", "I. Rate", "B. Price"]],
                                                    dfy["Price"], train_size = 0.65, random_state = rs)

x_train, x_test, y_train, y_test = train_test_split(dfy[["E. Rate", "I. Rate", "B. Price"]],
                                                    dfy["Line"], train_size = 0.65, random_state = rs)


from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
      
cosine_list = []
for i in np.arange(0,x_test.count()[0]):
    for j in np.arange(0,x_train.count()[0]):
        cos_vektor = cosine_similarity([x_test.values[i]], [x_train.values[j]])
        cosine_list.append(cos_vektor)

testvar_1 = cosine_list[0:21]
testvar_2 = cosine_list[21:42]
testvar_3 = cosine_list[42:63]
testvar_4 = cosine_list[63:84]
testvar_5 = cosine_list[84:105]
testvar_6 = cosine_list[105:126]
testvar_7 = cosine_list[126:147]
testvar_8 = cosine_list[147:168]
testvar_9 = cosine_list[168:189]
testvar_10 = cosine_list[189:210]
testvar_11 = cosine_list[210:231]
testvar_12 = cosine_list[231:252]

fullvar_list = [testvar_1, testvar_2, testvar_3, testvar_4, testvar_5, testvar_6,
                testvar_7, testvar_8, testvar_9, testvar_10, testvar_11, testvar_12]

t = 0
f = 0
tl = []

for j in np.arange(0,12):
    cos = pd.DataFrame()
    cos["Cosine Similarity"] = np.arange(0,21)
    cos["Index"] = np.arange(0,21)
    
    for i in np.arange(0,21):
        cos["Cosine Similarity"][i] = fullvar_list[j][i]
    print("------------------------------------------------------------------------")
    print(j+1,". Test Verisi")
    print(cos)
    
    testnum = j
    
    maxrow = cos[cos['Cosine Similarity']==cos['Cosine Similarity'].max()]
    maxrow_index = maxrow["Index"].values[0]

    print(x_test.values[testnum])
    print(x_train.values[maxrow_index])

    y_test.index = np.arange(0,12)
    print(y_test.iloc[testnum])
    y_train.index = np.arange(0,21)
    print(y_train.iloc[maxrow_index])


    if y_test.iloc[testnum] == y_train.iloc[maxrow_index]:
        t = t+1
    else:
        f = f+1
        
print(list(y_test_pred))

print("True Assignment  -->",t)
print("False Assignment -->",f)


#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

#As we mentioned we will perform a cosine similarity study and assign the curve that has highest score.
#We will use the test data (2022-10, 2023-06) in this section.

nx_train = df.drop("y", axis = 1)
ny_train = df["y"]
nx_test = df_test.drop("y", axis = 1)
ny_test = df_test["y"]



cosine_list = []
for i in np.arange(0,nx_test.count()[0]):
    for j in np.arange(0,nx_train.count()[0]):
        cos_vektor = cosine_similarity([nx_test.values[i]], [nx_train.values[j]])
        cosine_list.append(cos_vektor)
  
testvar_1 = cosine_list[0:33]
testvar_2 = cosine_list[33:66]
testvar_3 = cosine_list[66:99]
testvar_4 = cosine_list[99:132]
testvar_5 = cosine_list[132:165]
testvar_6 = cosine_list[165:198]
testvar_7 = cosine_list[198:231]
testvar_8 = cosine_list[231:264]
testvar_9 = cosine_list[264:297]


fullvar_list = [testvar_1, testvar_2, testvar_3, testvar_4, testvar_5, testvar_6,
                testvar_7, testvar_8, testvar_9]
line_list = []

for j in np.arange(0,9):
    cos = pd.DataFrame()
    cos["Cosine Similarity"] = np.arange(0,33)
    cos["Index"] = np.arange(0,33)
    cos["Line"] = np.arange(0,33)
    
    for i in np.arange(0,33):
        cos["Cosine Similarity"][i] = fullvar_list[j][i]
        cos["Line"][i] = dfy["Line"][i]
    print("------------------------------------------------------------------------")
    print(j+1,". Test Index")
    
    testnum = j
    
    maxrow = cos[cos['Cosine Similarity']==cos['Cosine Similarity'].max()]
    maxrow_index = maxrow["Index"].values[0]

    print(nx_test.values[testnum])
    print(nx_train.values[maxrow_index])


    ny_train.index = np.arange(0,33)
    print(cos["Line"].iloc[maxrow_index])
    line_list.append(cos["Line"].iloc[maxrow_index])


print(line_list)

#PREDICTION
#In this section, we will predict oil price values.
#All dependent variable values are found by substituting the 
#independent variables of that test data into the formula of the curve 
#that was previously assigned for each test data.

pred_list = []
print(df_test)

if K == 1:
    if M == 1:
        
        for i in np.arange(0,9):
            pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + 
                                    value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] +
                                    value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i])
   
            pred_list.append(pred)
            
    elif M == 2:
        
        for i in np.arange(0,9):
            pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + value(instance.c1[1,3]*df_test["x1"][i]**2) +
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] + value(instance.c2[1,3]*df_test["x2"][i]**2) +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i] + value(instance.c3[1,3]*df_test["x3"][i]**2))
                
            pred_list.append(pred)
        
    elif M == 3:
        
        for i in np.arange(0,9):
            pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + value(instance.c1[1,3]*df_test["x1"][i]**2) +
                                        value(instance.c1[1,4])*df_test["x1"][i]**3 +
                                 
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] + value(instance.c2[1,3]*df_test["x2"][i]**2) +
                                        value(instance.c2[1,4])*df_test["x2"][i]**3 +
                                 
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i] + value(instance.c3[1,3]*df_test["x3"][i]**2) +
                                        value(instance.c3[1,4])*df_test["x3"][i]**3)
            pred_list.append(pred)
    
elif K == 2:
    if M == 1:
        
        for i in np.arange(0,9):
            if line_list[i] == "First Line":
                pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + 
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i])
            elif line_list[i] == "Second Line":
                pred = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df_test["x1"][i] + 
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df_test["x2"][i] +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df_test["x3"][i])
            pred_list.append(pred)

    elif M == 2:

        for i in np.arange(0,9):
            if line_list[i] == "First Line":
                pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + value(instance.c1[1,3]*df_test["x1"][i]**2) +
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] + value(instance.c2[1,3]*df_test["x2"][i]**2) +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i] + value(instance.c3[1,3]*df_test["x3"][i]**2))
            elif line_list[i] == "Second Line":
                pred = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df_test["x1"][i] + value(instance.c1[2,3]*df_test["x1"][i]**2) +
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df_test["x2"][i] + value(instance.c2[2,3]*df_test["x2"][i]**2) +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df_test["x3"][i] + value(instance.c3[2,3]*df_test["x3"][i]**2))
            pred_list.append(pred)
            
    elif M == 3:

        for i in np.arange(0,9):
            if line_list[i] == "First Line":
                pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + value(instance.c1[1,3]*df_test["x1"][i]**2) +
                                        value(instance.c1[1,4])*df_test["x1"][i]**3 +
                                 
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] + value(instance.c2[1,3]*df_test["x2"][i]**2) +
                                        value(instance.c2[1,4])*df_test["x2"][i]**3 +
                                 
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i] + value(instance.c3[1,3]*df_test["x3"][i]**2) +
                                        value(instance.c3[1,4])*df_test["x3"][i]**3)
            elif line_list[i] == "Second Line":
                pred = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df_test["x1"][i] + value(instance.c1[2,3]*df_test["x1"][i]**2) +
                                        value(instance.c1[2,4])*df_test["x1"][i]**3 +
                                 
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df_test["x2"][i] + value(instance.c2[2,3]*df_test["x2"][i]**2) +
                                        value(instance.c2[2,4])*df_test["x2"][i]**3 +
                                 
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df_test["x3"][i] + value(instance.c3[2,3]*df_test["x3"][i]**2) +
                                        value(instance.c3[2,4])*df_test["x3"][i]**3)
            pred_list.append(pred)

elif K == 3:
    if M == 1:
        
        for i in np.arange(0,9):
            if line_list[i] == "First Line":
                pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + 
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i])
            elif line_list[i] == "Second Line":
                pred = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df_test["x1"][i] + 
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df_test["x2"][i] +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df_test["x3"][i])
            elif line_list[i] == "Third Line":
                pred = (value(instance.c1[3,1]) + value(instance.c1[3,2])*df_test["x1"][i] + 
                                        value(instance.c2[3,1]) + value(instance.c2[3,2])*df_test["x2"][i] +
                                        value(instance.c3[3,1]) + value(instance.c3[3,2])*df_test["x3"][i])
            pred_list.append(pred)
            
    elif M == 2:

        for i in np.arange(0,9):
            if line_list[i] == "First Line":
                pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + value(instance.c1[1,3]*df_test["x1"][i]**2) +
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] + value(instance.c2[1,3]*df_test["x2"][i]**2) +
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i] + value(instance.c3[1,3]*df_test["x3"][i]**2))
            elif line_list[i] == "Second Line":
                pred = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df_test["x1"][i] + value(instance.c1[2,3]*df_test["x1"][i]**2) +
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df_test["x2"][i] + value(instance.c2[2,3]*df_test["x2"][i]**2) +
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df_test["x3"][i] + value(instance.c3[2,3]*df_test["x3"][i]**2))
            elif line_list[i] == "Third Line":
                pred = (value(instance.c1[3,1]) + value(instance.c1[3,2])*df_test["x1"][i] + value(instance.c1[3,3]*df_test["x1"][i]**2) +
                                        value(instance.c2[3,1]) + value(instance.c2[3,2])*df_test["x2"][i] + value(instance.c2[3,3]*df_test["x2"][i]**2) +
                                        value(instance.c3[3,1]) + value(instance.c3[3,2])*df_test["x3"][i] + value(instance.c3[3,3]*df_test["x3"][i]**2))
            pred_list.append(pred)
            
    elif M == 3:
        
        for i in np.arange(0,9):
            if line_list[i] == "First Line":
                pred = (value(instance.c1[1,1]) + value(instance.c1[1,2])*df_test["x1"][i] + value(instance.c1[1,3]*df_test["x1"][i]**2) +
                                        value(instance.c1[1,4])*df_test["x1"][i]**3 +
                                 
                                        value(instance.c2[1,1]) + value(instance.c2[1,2])*df_test["x2"][i] + value(instance.c2[1,3]*df_test["x2"][i]**2) +
                                        value(instance.c2[1,4])*df_test["x2"][i]**3 +
                                 
                                        value(instance.c3[1,1]) + value(instance.c3[1,2])*df_test["x3"][i] + value(instance.c3[1,3]*df_test["x3"][i]**2) +
                                        value(instance.c3[1,4])*df_test["x3"][i]**3)
            elif line_list[i] == "Second Line":
                pred = (value(instance.c1[2,1]) + value(instance.c1[2,2])*df_test["x1"][i] + value(instance.c1[2,3]*df_test["x1"][i]**2) +
                                        value(instance.c1[2,4])*df_test["x1"][i]**3 +
                                 
                                        value(instance.c2[2,1]) + value(instance.c2[2,2])*df_test["x2"][i] + value(instance.c2[2,3]*df_test["x2"][i]**2) +
                                        value(instance.c2[2,4])*df_test["x2"][i]**3 +
                                 
                                        value(instance.c3[2,1]) + value(instance.c3[2,2])*df_test["x3"][i] + value(instance.c3[2,3]*df_test["x3"][i]**2) +
                                        value(instance.c3[2,4])*df_test["x3"][i]**3)
            elif line_list[i] == "Third Line":
                pred = (value(instance.c1[3,1]) + value(instance.c1[3,2])*df_test["x1"][i] + value(instance.c1[3,3]*df_test["x1"][i]**2) +
                                        value(instance.c1[3,4])*df_test["x1"][i]**3 +
                                 
                                        value(instance.c2[3,1]) + value(instance.c2[3,2])*df_test["x2"][i] + value(instance.c2[3,3]*df_test["x2"][i]**2) +
                                        value(instance.c2[3,4])*df_test["x2"][i]**3 +
                                 
                                        value(instance.c3[3,1]) + value(instance.c3[3,2])*df_test["x3"][i] + value(instance.c3[3,3]*df_test["x3"][i]**2) +
                                        value(instance.c3[3,4])*df_test["x3"][i]**3)
            pred_list.append(pred)


print(pred_list)
print(list(df_test["y"]))

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

#Finally, a graph that shows us prediction values and real oil price values 
#will be created and MAE will be calculated.


x = np.arange('2022-10', '2023-07', dtype='datetime64[M]')

plt.figure(figsize = (8,5))
plt.plot(x, pred_list, color = "black")
plt.plot(x, list(df_test["y"]), color = "black")
plt.scatter(x, pred_list, color = "red", s = 100, zorder = 2)
plt.scatter(x, list(df_test["y"]), color = "green", s = 100, zorder = 2)
plt.xticks( fontsize=12, fontweight='bold')
plt.yticks( fontsize=12, fontweight='bold')
plt.grid()
plt.tight_layout()
plt.ylim(10,30)
plt.gcf().autofmt_xdate()
plt.xlabel("DATE", fontsize=12, fontweight='bold')
plt.ylabel("OIL PRICE", fontsize=12, fontweight='bold')
plt.show()

#MAE

mae_list = []
for i in np.arange(0,9):
    mae_list.append(np.absolute(pred_list[i] - list(df_test["y"])[i]))

mae = sum(mae_list) / 9 

print("MAE =",mae)







