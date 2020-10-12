#!/usr/bin/env python
# MSM2018-AtomSims-HW, MC simulation of LJ particles
# Xiaohong Zhang
# Getman Research Group
# 03/23/2018

# To use:
# python mc.py -n 100 -b 15 -t 1.0 -r 10000  -l ps 


import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

np.random.seed(379245)

atom = {'Ar':{'mass':39.948,'eps':1.65e-21,'sigma':3.4e-10}} 
a_name = 'Ar'
m = atom[a_name]['mass'] *1.6747e-27                # mass, [kg]
eps = atom[a_name]['eps']                           # epsilon, [J]
sigma = atom[a_name]['sigma']                       # sigma in Lennard-Jones Potential, [m]
kB = 1.380e-23                                      # Boltzmann, [J/K]
Na = 6.022e23                                       # Avogadro number                                          
max_disp = 1.5                                      # max displacement of MC move

rc = 2.5                                            # LJ cutoff distance
rc2 = rc**2
rci = 1/rc
rc6i = rci**6
rs = 2.0                                            # potential switch distance
rs2= rs**2                        
ec = 4.0*rc6i*(rc6i - 1)                            # LJ cutoff energy



def initUnifPos(N,L):
    """ generate simulation box filled with N particles and inititial positions """
    nc = int(np.floor(np.cbrt(N)))                  # cut box into small cubic cells
    lc = np.float128(L/nc)                          # length of each small cubic cell
    pos_arr = np.zeros(shape=(N,3))                 # store coordinates 3*[x,y,z]
    f = open('init_pos.xyz','w')
    f.write(str(N) + "\n" + "LJ particles\n")       # write number of particles
    
    for i in range(int(N//np.power(nc,3))+1):       # loop over big box, refill from start after first round filling (nc^3)
        for xx in range(nc):                        # loop over x direction of the big box
            for yy in range(nc):                    # loop over y direction of the big box
                for zz in range(nc):                # loop over z direction of the big box
                    
                    idx = i*np.power(nc,3) + xx*np.power(nc,2) + yy*nc + zz
                    
                    if (i==1):                      # next round filling
                        if (idx >= N):              # stop filling particle when exceeding given particle number
                            break
                            
                    x = '{:.12e}'.format(np.random.uniform(low=xx*lc+0.1*lc, high=(xx+1)*lc-0.1*lc)) 
                    y = '{:.12e}'.format(np.random.uniform(low=yy*lc+0.1*lc, high=(yy+1)*lc-0.1*lc)) 
                    z = '{:.12e}'.format(np.random.uniform(low=zz*lc+0.1*lc, high=(zz+1)*lc-0.1*lc))                               
                                
                    pos_arr[idx] = [x,y,z]
                    
                    p = "Ar " + str(x) + " " + str(y) + " " + str(z) + "\n"              
                    f.write(str(p))                                        
    f.close()
                                         
    return pos_arr      




def wrapPBC(pos_arr,N,L):
    """ wrap the coordinates, use in trajectory """
    for i in range(N):   
        pos_arr[i] = pos_arr[i] - L*np.floor(pos_arr[i]/L)
    
    return pos_arr



def getLJ(pos_arr,ljTail,N,L): 
    
    en = 0.0    
    r_vec = np.zeros(shape=(1,3))
    
    for i in range(N-1):
        for j in range(i+1, N):
            
            r_vec = pos_arr[i] - pos_arr[j]         # r_vec=[rx,ry,rz]
            r_vec = r_vec - L*np.rint(r_vec/L)      # PBC condition            
            r2 = np.sum(r_vec**2)               

            if (r2 <= rc2):
                r2i = 1/r2
                r6i = r2i**3
                
                if (ljTail == "trunc"):             # trunc at cutoff
                    ee = 4*r6i*(r6i-1)    

                elif (ljTail == "ps"):              # potential shift
                    ee = 4*r6i*(r6i-1) - ec   
                                        
                elif (ljTail == "fs"):              # force shift      
                    r = np.sqrt(r2)
                    ee = 4*r6i*(r6i-1) - ec + 48*rci*rc6i*(rc6i-0.5)*(r-rc)

                elif (ljTail == "psw"):             # potential switch 
                    r = np.sqrt(r2)
                    if (r < rs):
                        s = 1
                        ee = s*4*r6i*(r6i-1)
                        
                    elif (r >= rs):              
                        R = (r2 - rs2)/(rc2 - rs2)  # use in switch function 
                        s = 1 + R**2 * (2*R - 3)
                        ee = s*4*r6i*(r6i-1)                 
                
                en = en + ee       
    return en



def getLJpartial(pos_arr,rnd_idx,ljTail,N,L):       # for energy change
    
    en = 0.0 
    r_vec = np.zeros(shape=(1,3))
    
    for i in range(N):
        if not (rnd_idx == i):
            
            r_vec = pos_arr[i] - pos_arr[rnd_idx]     
            r_vec = r_vec - L*np.rint(r_vec/L)         
            r2 = np.sum(r_vec**2)               
    
            if (r2 <= rc2):
                r2i = 1/r2
                r6i = r2i**3
                
                if (ljTail == "trunc"):   
                    ee = 4*r6i*(r6i-1)    

                elif (ljTail == "ps"):    
                    ee = 4*r6i*(r6i-1) - ec   
                                        
                elif (ljTail == "fs"): 
                    r = np.sqrt(r2)
                    ee = 4*r6i*(r6i-1) - ec + 48*rci*rc6i*(rc6i-0.5)*(r-rc)

                elif (ljTail == "psw"):      
                    r = np.sqrt(r2)
                    if (r < rs):
                        s = 1
                        ee = s*4*r6i*(r6i-1)
                        
                    elif (r >= rs):              
                        R = (r2 - rs2)/(rc2 - rs2)   
                        s = 1 + R**2 * (2*R - 3)
                        ee = s*4*r6i*(r6i-1)                                 
                
                en = en + ee         
    return en



def writeCoords(r_c,N,L):

    r_wrap = wrapPBC(r_c,N,L)
    one_traj = []
    one_traj.append(str(N) + "\n" + "LJ particles\n")    
    for i in range(N):                               
        x = '{:.7f}'.format(r_wrap[i][0]) 
        y = '{:.7f}'.format(r_wrap[i][1]) 
        z = '{:.7f}'.format(r_wrap[i][2])   
        p = "Ar " + str(x) + " " + str(y) + " " + str(z) + "\n"             
        one_traj.append(str(p))  
        
    return one_traj



def writeThermoVal(thermo_val,N):
        
    thermo_val[:,1:] = thermo_val[:,1:]*eps*Na/N*1.0e-3   # convert to [kJ/mol]
    np.savetxt('thermoVal.txt', thermo_val, fmt='%8.3f',header='step,   en   avg_en  in (kJ/mol)')  

    return thermo_val



def plotThermo(thermo_val):
    
    fig, ax = plt.subplots()
    ax.set_title('MC energy')
    ax.plot(thermo_val[:,0], thermo_val[:,1], label='EP')
    ax.plot(thermo_val[:,0], thermo_val[:,2], label='EP running avg ')
    ax.set_title('MC energy')
    plt.xlabel('Steps')
    plt.ylabel('Energy (kJ/mol)')
    plt.legend() 
    plt.grid(True)
#    plt.savefig('mcThermoVal.png') 
    plt.show()       


    
def main():   

    
    parser = argparse.ArgumentParser(description='Parameters for running LJ MD program.')
    parser.add_argument('--partNumber', '-n', default=100, type=int, help='Number of particles. Default is 100')
    parser.add_argument('--boxLength', '-b', default=15.0, type=float, help='Length of simulation box. Default is 15.0')
    parser.add_argument('--temperature', '-t', default=1.0, type=float, help='NVT temperature. Default is 1.0')
    parser.add_argument('--runSteps', '-r', default=10000, type=int, help='MC simulation run steps. Default is 10000')                          
    parser.add_argument('--ljTail', '-l', default='ps', choices=['trunc','ps','fs','psw'],
                        type=str.lower,help='LJ potential tail correction method, default is psw,\
                        trunc--truncate LJ at cutoff, ps--potential shift,fs--force shift, psw--potential switch.')    
    
    args = parser.parse_args()
    N = args.partNumber
    L = args.boxLength
    T = args.temperature
    nSteps = args.runSteps
    ljTail = args.ljTail    
    beta = 1.0/T
    print ("\nMC simulation of Argon particles with LJ potential\n")
    print ("Particle number: %s\n" % N)
    print ("Box length: %.1f (sigma)\n" % L)
    print ("Argon density: %.3f kg/m^3\n" % (m*N/np.power(L*sigma,3)))


    r_c = initUnifPos(N,L)                    # get initial position
    en = getLJ(r_c,ljTail,N,L)
    mcTraj = [] 
    one_traj = writeCoords(r_c,N,L)           # write trajectory
    mcTraj.append(one_traj)
    
    step = 0                                  # start at step 0      
    total_move = 0
    accepted_move = 0
    sum_en = 0
    thermo_val = []                           # save thermo values
    r_trial = np.zeros(shape=(N,3))           # trial position

    
    for step in range(nSteps):  

        if (step%1000 == 0):
            print ("steps: ", step)
            
        rnd_idx = np.random.randint(N-1)       
        r_trial = np.copy(r_c)
        r_trial[rnd_idx] = r_trial[rnd_idx] + max_disp*(np.random.uniform(size=(1,3)) - 0.5)
        r_trial = wrapPBC(r_trial,N,L)
        delta_lj = getLJpartial(r_trial,rnd_idx,ljTail,N,L) - getLJpartial(r_c,rnd_idx,ljTail,N,L)
        
        if (np.random.uniform() < np.exp(-beta*delta_lj)):
            r_c = r_trial
            accepted_move += 1
            en = en + delta_lj
            
        total_move += 1
        sum_en += en
        avg_en = sum_en/total_move        
                        
        thermo_val.append([step, en, avg_en])        
        one_traj = writeCoords(r_c,N,L) 
        mcTraj.append(one_traj)
            
    print("acceptance ratio: %.1f" % (float(accepted_move)/total_move))      
    
    thermo_val = np.asarray(thermo_val)
    thermo_val = writeThermoVal(thermo_val,N)
    file = open('mcTraj.xyz','w')  
    for i in range(len(mcTraj)):
        for j in range(len(mcTraj[i])):
            file.write(mcTraj[i][j])
    file.close()
    plotThermo(thermo_val)

        
if __name__ == "__main__":
    start_time = time.time()
    main()            
    print("--- Runtime: %.2f seconds ---" % (time.time() - start_time))        