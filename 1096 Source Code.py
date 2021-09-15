#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:31:02 2020

@author: ericchen
"""

def swing(y, t, m, M, g, d_prime2, d, b):
    import numpy as np
    phi, omega = y
    dydt = [omega, (-m*g*d*np.sin(phi)-b*omega)/((m+M)*d_prime2)]
    return dydt

    
def d_prime2_func(d):
    m = 0.5
    M = 1.2
    beta = 0.9
    return (beta*M*(0.4**2)+m*(d**2))/(m+M)



def alt_rotations():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    m = 0.5
    M= 1.2
    g = 9.8
    d_out = 0.5
    d_in = 0.4
    b = 0.0525
    n = 50
    
    d = 0.0
    previous_phi = 5*np.pi/8
    previous_omega = 0.0
    diverging = False
    
    all_phi = np.array([0])
    all_phidot = np.array([0])
    all_time = np.array([0])
    
    for j in range(0,n):
        for k in range (0,4):
            
            #k=0
            flag = True
            if k%2 == 0:
                y0 = [previous_phi, previous_omega]
                d = d_out
                d_prime2 = d_prime2_func(d)
                flag = True
            elif k%2 == 1:
                d_prev = d
                d = d_in
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                y0 = [0.0, omega]
                flag = False
                
            t = np.linspace(0,2,400)
            sol = odeint(swing, y0, t, args = (m,M,g,d_prime2,d,b))
            sol = np.array(sol)
            sol_abs = np.abs(sol)
            
            index = 0
                
            if flag == True:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,0]>0:
                        if sol[l,0]>0 and sol[l+1,0]<0:
                            index = l
                            break
                    elif sol[0,0]<0:
                        if sol[l,0]<0 and sol[l+1,0]>0:
                            index = l
                            break
                previous_phi = 0.0
                previous_omega = sol[index, 1]
                
            elif flag == False:                    
                if sol_abs[:,1].min(axis=0)<0.1:
                    #index = np.where(sol_abs[:,1]==sol_abs[:,1].min(axis=0))
                    #previous_phi = sol[index,0]
                    #previous_omega = 0.0
                    for o in range(0,len(sol)-1):
                        if sol[0,1]>0:
                            if sol[o,1]>0 and sol[o+1,1]<0:
                                index = o
                                break
                        elif sol[0,1]<0:
                            if sol[o,1]<0 and sol[o+1,1]>0:
                                index = o
                                break
                    previous_phi = sol[index,0]
                    previous_omega = 0.0
                            
                else:
                    diverging = True
                    for p in range (0,len(sol)-1):
                        if sol[0,1]>0:
                            if sol[p,0]-np.pi<0 and sol[p+1,0]-np.pi>0:
                                index = p
                                previous_phi = -np.pi
                                break
                        elif sol[0,1]<0:
                            if sol[p,0]+np.pi>0 and sol[p+1,0]+np.pi<0:
                                index = p
                                previous_phi = np.pi
                                break
                    previous_omega = sol[index, 1]
                    
                
            #index = int(''.join(map(str,index[0])))
            plt.plot(sol[0:index,0], sol[0:index,1], 'b')
            plt.xlabel('Phi(rad)')
            plt.ylabel('Phi_dot(rad/s)')
            
            all_phi = np.append(all_phi,sol[0:index,0])
            all_phidot = np.append(all_phidot,sol[0:index,1])
            all_time = np.append(all_time,t[0:index]+all_time[-1])
            #k=k+1
            
            if diverging ==True:
                break
    
        if diverging == True: 
            spinning(previous_phi, previous_omega, m,M,g,b,d_out,d_in, n, all_phi, all_phidot, all_time)
            break
        
    plt.grid()    
    plt.show()
    
            

def spinning(phi_i, omega_i, m, M, g, b, d_out, d_in, n, allp, allpd, allt):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    t_p = 0
    m = m
    M= M
    g = g
    b = b
    d_out = d_out
    d_in = d_in
    
    all_phi = allp
    all_phidot = allpd
    all_time = allt
    
    #phi_i = np.pi
    #omega_i = -0.52478
    
    previous_phi = phi_i
    previous_omega = omega_i
    d = 0.0
    
    #decide the color gradient
    gradient = int(n/5)
    grad_a = np.linspace(0.7,0,4*gradient)
    grad_b = np.linspace(1,0.8, gradient + n%5)
    
    
    for j in range(0,n):
        for k in range (0,2):
            #k=0
            flag = True
            if k%2 == 0:
                d_prev = d_in
                d = d_out
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                #print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
                flag = True
            elif k%2 == 1:
                d_prev = d_out
                d = d_in
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                #print(d_prime2_func(d_prev)/d_prime2)
                y0 = [0.0, omega]
                flag = False
                
            t = np.linspace(0,5,1000)
            sol = odeint(swing, y0, t, args = (m,M,g,d_prime2,d,b))
            sol = np.array(sol)
            
            index = 0
                
            if flag == True:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,0]>0:
                        if sol[l,0]>0 and sol[l+1,0]<0:
                            index = l
                            break
                    elif sol[0,0]<0:
                        if sol[l,0]<0 and sol[l+1,0]>0:
                            index = l
                            break
                previous_phi = 0.0
                previous_omega = sol[index, 1]
                
            elif flag == False:
                    #index = np.where(sol_abs[:,1]==sol_abs[:,1].min(axis=0))
                    #previous_phi = sol[index,0]
                    #previous_omega = 0.0
                for o in range(0,len(sol)-1):
                    if sol[0,1]>0:
                        if sol[o,0]-np.pi<0 and sol[o+1,0]-np.pi>0:
                            index = o
                            previous_phi = -np.pi
                            break
                    elif sol[0,1]<0:
                        if sol[o,0]+np.pi>0 and sol[o+1,0]+np.pi<0:
                            index = o
                            previous_phi = np.pi
                            break
                previous_omega = sol[index, 1]
                            
                
            #index = int(''.join(map(str,index[0])))
            if (j>=4*gradient):
                plt.plot(sol[0:index,0], sol[0:index,1], color=(grad_b[j-4*gradient],0,0))
            else:
                plt.plot(sol[0:index,0], sol[0:index,1], color=(1,grad_a[j],grad_a[j]))

            plt.xlabel('Phi(rad)')
            plt.ylabel('Phi_dot(rad/s)')
            #k=k+1
            
            all_phi = np.append(all_phi,sol[0:index,0])
            all_phidot = np.append(all_phidot,sol[0:index,1])
            all_time = np.append(all_time, t[0:index]+all_time[-1])
            
            if (j==n-1):
                t_p = t_p + t[index]
                

    plt.grid()       
    plt.show()

    plt.plot(all_time[1:len(all_time)-1], all_phi[1:len(all_phi)-1], color = 'g', label='phi(t)')   
    plt.plot(all_time[1:len(all_time)-1], all_phidot[1:len(all_phidot)-1], color = 'b', label = 'omega(t)')
    plt.xlabel('Time(s)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    print (np.abs(all_phidot).max(axis=0))
    print (t_p)
    
import numpy as np
all_phi = np.array([0])
all_phidot = np.array([0])
all_time = np.array([0])
spinning(-np.pi, 0.001,0.5, 1.2, 9.8, 0.0525, 0.5, 0.4, 50, all_phi, all_phidot, all_time)    
alt_rotations()


# m = 0.5 kg
# M = 1.2kg
# b = 
# beta = 
# d_out = 
# d_in = 





def tangent_swing(y, t, beta, R, m, M, g, d, b):
    import numpy as np
    phi, omega = y
    dydt = [omega, (-m*g*R*np.sin(phi+(d/R))-b*omega)/((m+beta*M)*R**2)]
    return dydt

def tangent(phi_i, omega_i, beta, R, m, M, g, b, d0, n, allp, allpd, allt):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    

    m = m
    M= M
    g = g
    b = b
    
    all_phi = allp
    all_phidot = allpd
    all_time = allt
    
    #phi_i = np.pi
    #omega_i = -0.52478
    
    previous_phi = phi_i
    previous_omega = omega_i
    d = 0.0
    
    #decide the color gradient
    
    diverging = False
    
    for j in range(0,n):
        for k in range (0,6):
            #k=0
            if k %2 == 0:
                d = -d0/2
            elif k%2 == 1:
                d = d0/2
                
            y0 = [previous_phi, previous_omega]
                
            t = np.linspace(0,2,400)
            sol = odeint(tangent_swing, y0, t, args = (beta, R, m, M, g, d, b))
            sol = np.array(sol)
            sol_abs = np.abs(sol)
            
            index = 0
                
            if k==0 or k ==5:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,1]>0:
                        if sol_abs[:,1].min(axis=0)<0.1:
                            if sol[l,1]>0 and sol[l+1,1]<0:
                                index = l
                                previous_phi = sol[index, 0]
                                previous_omega = 0.0
                                break
                        else:
                            diverging = True
                            if sol[l,0]-np.pi<0 and sol[l+1,0]-np.pi>0:
                                index = l
                                previous_phi = -np.pi
                                previous_omega = sol[index,1]
                                break
                            
                    elif sol[1,1]<0:
                        if sol[l,0]-(np.pi/2)>0 and sol[l+1,0]-(np.pi/2)<0:
                            index = l
                            previous_phi = np.pi/2
                            previous_omega = sol[index,1]
                            break
            
                
            elif k==1 or k==4:
                    #index = np.where(sol_abs[:,1]==sol_abs[:,1].min(axis=0))
                    #previous_phi = sol[index,0]
                    #previous_omega = 0.0
                for o in range(0,len(sol)-1):
                    if sol[0,1]<0:
                        if sol[o,0]+(np.pi/2)>0 and sol[o+1,0]+(np.pi/2)<0:
                            index = o
                            previous_phi = -np.pi/2
                            break
                    elif sol[0,1]>0:
                        if sol[o,0]-(np.pi/2)<0 and sol[o+1,0]-(np.pi/2)>0:
                            index = o
                            previous_phi = np.pi/2
                            break
                previous_omega = sol[index, 1]
            
            elif k==2 or k ==3:
                for p in range(0, len(sol)-1):
                    if sol[0,1]<0:
                        if sol_abs[:,1].min(axis=0)<0.1:
                            if sol[p,1]<0 and sol[p+1,1]>0:
                                index = p
                                previous_phi = sol[index, 0]
                                previous_omega = 0.0
                                break
                        else:
                            diverging = True
                            if sol[p,0]+np.pi>0 and sol[p+1,0]+np.pi<0:
                                index = p
                                previous_phi = np.pi
                                previous_omega = sol[index,1]
                                break
                    elif sol[1,1]>0:
                        if sol[p,0]+(np.pi/2)<0 and sol[p+1,0]+(np.pi/2)>0:
                            index = p
                            previous_phi = -np.pi/2
                            previous_omega = sol[index,1]
                            break
                            
                
            #index = int(''.join(map(str,index[0])))
            plt.plot(sol[0:index,0], sol[0:index,1], 'b')
            plt.xlabel('Phi(rad)')
            plt.ylabel('Phi_dot(rad/s)')
            #k=k+1
            
            all_phi = np.append(all_phi,sol[0:index,0])
            all_phidot = np.append(all_phidot,sol[0:index,1])
            all_time = np.append(all_time, t[0:index]+all_time[-1])
            
            if diverging ==True:
                break
           
        if diverging == True: 
            spinning_tangent(previous_phi, previous_omega,beta, R, m,M,g,b,d0,n, all_phi, all_phidot, all_time)
            break     
        
    plt.grid        
    plt.show()

def spinning_tangent(phi_i, omega_i, beta, R, m, M, g, b, d0, n, allp, allpd, allt):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    t_p = 0
    m = m
    M= M
    g = g
    b = b
    
    all_phi = allp
    all_phidot = allpd
    all_time = allt
    
    #phi_i = np.pi
    #omega_i = -0.52478
    x = 0
    previous_phi = phi_i
    previous_omega = omega_i
    
    if omega_i>0:
        x = 3
        y = 6
    else:
        x = 0
        y = 3
        
    d = 0.0
    
    #decide the color gradient
    gradient = int(n/5)
    grad_a = np.linspace(0.7,0,4*gradient)
    grad_b = np.linspace(1,0.8, gradient + n%5)
        
    
    for j in range(0,n):
        for k in range (x,y):
            #k=0
            if k %2 == 0:
                d = -d0/2
            elif k%2 == 1:
                d = d0/2
                
            y0 = [previous_phi, previous_omega]
                
            t = np.linspace(0,4,800)
            sol = odeint(tangent_swing, y0, t, args = (beta, R, m, M, g, d, b))
            sol = np.array(sol)
            
            index = 0
                
            if k==0 or k ==5:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,1]>0:
                        if sol[l,0]-np.pi<0 and sol[l+1,0]-np.pi>0:
                                index = l
                                previous_phi = -np.pi
                                previous_omega = sol[index,1]
                                break
                        
                    elif sol[0,1]<0:
                        if sol[l,0]-(np.pi/2)>0 and sol[l+1,0]-(np.pi/2)<0:
                            index = l
                            previous_phi = np.pi/2
                            previous_omega = sol[index,1]
                            break
            
                
            elif k==1 or k==4:
                    #index = np.where(sol_abs[:,1]==sol_abs[:,1].min(axis=0))
                    #previous_phi = sol[index,0]
                    #previous_omega = 0.0
                for o in range(0,len(sol)-1):
                    if sol[0,1]<0:
                        if sol[o,0]+(np.pi/2)>0 and sol[o+1,0]+(np.pi/2)<0:
                            index = o
                            previous_phi = -np.pi/2
                            break
                    elif sol[0,1]>0:
                        if sol[o,0]-(np.pi/2)<0 and sol[o+1,0]-(np.pi/2)>0:
                            index = o
                            previous_phi = np.pi/2
                            break
                previous_omega = sol[index, 1]
            
            elif k==2 or k ==3:
                for p in range(0, len(sol)-1):
                    if sol[0,1]<0:
                        if sol[p,0]+np.pi>0 and sol[p+1,0]+np.pi<0:
                            index = p
                            previous_phi = np.pi
                            previous_omega = sol[index,1]
                            break
                            
                    elif sol[0,1]>0:
                        if sol[p,0]+(np.pi/2)<0 and sol[p+1,0]+(np.pi/2)>0:
                            index = p
                            previous_phi = -np.pi/2
                            previous_omega = sol[index,1]
                            break
                            
                
            if (j>=4*gradient):
                plt.plot(sol[0:index,0], sol[0:index,1], color=(grad_b[j-4*gradient],0,0))
            else:
                plt.plot(sol[0:index,0], sol[0:index,1], color=(1,grad_a[j],grad_a[j]))

            plt.xlabel('Phi(rad)')
            plt.ylabel('Phi_dot(rad/s)')
            #k=k+1
            
            all_phi = np.append(all_phi,sol[0:index,0])
            all_phidot = np.append(all_phidot,sol[0:index,1])
            all_time = np.append(all_time, t[0:index]+all_time[-1])
            
            if j==n-1:
                t_p = t_p + t[index]
    plt.grid        
    plt.show()

    plt.plot(all_time[1:len(all_time)-1], all_phi[1:len(all_phi)-1], color = 'g', label='phi(t)')   
    plt.plot(all_time[1:len(all_time)-1], all_phidot[1:len(all_phidot)-1], color = 'b', label = 'omega(t)')
    plt.xlabel('Time(s)')
    plt.legend(loc='best')
    plt.grid
    plt.show()
    print (np.abs(all_phidot).max(axis=0))
    print (t_p)

all_phi = np.array([0])
all_phidot = np.array([0])
all_time = np.array([0])
tangent(5*np.pi/8,0.0,0.9,0.4,0.5,1.2,9.8,0.0525,0.15,50,all_phi,all_phidot,all_time)
spinning_tangent(-np.pi,0.003,0.9,0.4,0.5,1.2,9.8,0.0525,0.15,30,all_phi,all_phidot,all_time)

#


def hybrid_swing(y, t, beta, R, m, M, g, d_rad, d_tan, b, d_prime2):
    import numpy as np
    phi, omega = y
    dydt = [omega, (-m*g*d_rad*np.sin(phi+(d_tan/R))-b*omega)/((m+m)*d_prime2)]
    return dydt

def hybrid(phi_i, omega_i, beta, R, m, M, g, b, d0, d_in, d_out, n, allp, allpd, allt):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    m = m
    M= M
    g = g
    b = b
    R = R
    
    all_phi = allp
    all_phidot = allpd
    all_time = allt
    
    #phi_i = np.pi
    #omega_i = -0.52478
    
    previous_phi = phi_i
    previous_omega = omega_i
    d = 0.0
    d_tan = 0.0
    
    #decide the color gradient
    
    diverging = False
    
    
    for j in range(0,n):
        for k in range (0,8):
            #k=0
            if k == 0:
                d_tan = -d0/2
                d_prev = d_in
                d = d_out
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                #print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 1:
                d_tan = d0/2
                y0 = [previous_phi, previous_omega]
            elif k == 2:
                d_tan = d0/2
                d_prev = d_out
                d = d_in
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                #print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 3:
                d_tan = -d0/2
                y0 = [previous_phi, previous_omega]
            elif k == 4:
                d_tan = d0/2
                d_prev = d_in
                d = d_out
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                #print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 5:
                d_tan = -d0/2
                y0 = [previous_phi, previous_omega]
            elif k == 6:
                d_tan = -d0/2
                d_prev = d_out
                d = d_in
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                #print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 7:
                d_tan = d0/2
                y0 = [previous_phi, previous_omega]

                            
            t = np.linspace(0,2,400)
            sol = odeint(hybrid_swing, y0, t, args = (beta, R, m, M, g, d, d_tan, b, d_prime2))
            sol = np.array(sol)
            sol_abs = np.abs(sol)
            
            index = 0
                
            if k==0 or k ==7:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,1]>0:
                        if sol_abs[:,1].min(axis=0)>0.1 or (sol[l,0]-np.pi<0 and sol[l+1,0]-np.pi>0):
                            if sol[l,0]-np.pi<0 and sol[l+1,0]-np.pi>0:
                                diverging = True
                                index = l
                                previous_phi = -np.pi
                                previous_omega = sol[index,1]
                                break
                        else:
                            if sol[l,1]>0 and sol[l+1,1]<0:
                                index = l
                                previous_phi = sol[index, 0]
                                previous_omega = 0.0
                                break
                            
                    elif sol[1,1]<0:
                        if sol[l,0]-(np.pi/2)>0 and sol[l+1,0]-(np.pi/2)<0:
                            index = l
                            previous_phi = np.pi/2
                            previous_omega = sol[index,1]
                            break
            
                
            elif k==2 or k==6:
                    #index = np.where(sol_abs[:,1]==sol_abs[:,1].min(axis=0))
                    #previous_phi = sol[index,0]
                    #previous_omega = 0.0
                for o in range(0,len(sol)-1):
                    if sol[0,1]<0:
                        if sol[o,0]+(np.pi/2)>0 and sol[o+1,0]+(np.pi/2)<0:
                            index = o
                            previous_phi = -np.pi/2
                            break
                    elif sol[0,1]>0:
                        if sol[o,0]-(np.pi/2)<0 and sol[o+1,0]-(np.pi/2)>0:
                            index = o
                            previous_phi = np.pi/2
                            break
                previous_omega = sol[index, 1]
                
            elif k==1 or k==5:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,0]>0:
                        if sol[l,0]>0 and sol[l+1,0]<0:
                            index = l
                            break
                    elif sol[0,0]<0:
                        if sol[l,0]<0 and sol[l+1,0]>0:
                            index = l
                            break
                previous_phi = 0.0
                previous_omega = sol[index, 1]
            
            elif k==3 or k ==4:
                for p in range(0, len(sol)-1):
                    if sol[0,1]<0:
                        if sol_abs[:,1].min(axis=0)>0.1 or (sol[p,0]+np.pi>0 and sol[p+1,0]+np.pi<0):
                            if sol[p,0]+np.pi>0 and sol[p+1,0]+np.pi<0:
                                diverging = True
                                index = p
                                previous_phi = np.pi
                                previous_omega = sol[index,1]
                                break
                        else:
                            if sol[p,1]<0 and sol[p+1,1]>0:
                                index = p
                                previous_phi = sol[index, 0]
                                previous_omega = 0.0
                                break
                            
                    elif sol[1,1]>0:
                        if sol[p,0]+(np.pi/2)<0 and sol[p+1,0]+(np.pi/2)>0:
                            index = p
                            previous_phi = -np.pi/2
                            previous_omega = sol[index,1]
                            break
                            
                
            #index = int(''.join(map(str,index[0])))
            plt.plot(sol[0:index,0], sol[0:index,1], 'b')
            plt.xlabel('Phi(rad)')
            plt.ylabel('Phi_dot(rad/s)')
            #k=k+1
            
            all_phi = np.append(all_phi,sol[0:index,0])
            all_phidot = np.append(all_phidot,sol[0:index,1])
            all_time = np.append(all_time, t[0:index]+all_time[-1])
            
            if diverging ==True:
                break
           
        if diverging == True: 
            spinning_hybrid(previous_phi, previous_omega,beta, R, m,M,g,b,d0,d_in, d_out,n, all_phi, all_phidot, all_time)
            break     
        
    plt.grid()    
    plt.show()
    plt.plot(all_time[1:len(all_time)-1], all_phi[1:len(all_phi)-1], color = 'g', label='phi(t)')   
    plt.plot(all_time[1:len(all_time)-1], all_phidot[1:len(all_phidot)-1], color = 'b', label = 'omega(t)')
    plt.xlabel('Time(s)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def spinning_hybrid(phi_i, omega_i, beta, R, m, M, g, b, d0, d_in, d_out, n, allp, allpd, allt):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    

    m = m
    M= M
    g = g
    b = b
    t_p = 0
    
    all_phi = allp
    all_phidot = allpd
    all_time = allt
    
    #phi_i = np.pi
    #omega_i = -0.52478
    x = 0
    y = 0
    previous_phi = phi_i
    previous_omega = omega_i
    
    if omega_i>0:
        x = 4
        y = 8
    else:
        x = 0
        y = 4
        
    d = 0.0
    
    #decide the color gradient
    gradient = int(n/5)
    grad_a = np.linspace(0.7,0,4*gradient)
    grad_b = np.linspace(1,0.8, gradient + n%5)
        
    
    for j in range(0,n):
        for k in range (x,y):
            #k=0
            if k == 0:
                d_tan = -d0/2
                d_prev = d_in
                d = d_out
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 1:
                d_tan = d0/2
                y0 = [previous_phi, previous_omega]
            elif k == 2:
                d_tan = d0/2
                d_prev = d_out
                d = d_in
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 3:
                d_tan = -d0/2
                y0 = [previous_phi, previous_omega]
            elif k == 4:
                d_tan = d0/2
                d_prev = d_in
                d = d_out
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 5:
                d_tan = -d0/2
                y0 = [previous_phi, previous_omega]
            elif k == 6:
                d_tan = -d0/2
                d_prev = d_out
                d = d_in
                d_prime2 = d_prime2_func(d)
                omega = (d_prime2_func(d_prev)/d_prime2)*previous_omega
                print(d_prime2_func(d_prev)/d_prime2)
                y0 = [previous_phi, omega]
            elif k == 7:
                d_tan = d0/2
                y0 = [previous_phi, previous_omega]

                            
            t = np.linspace(0,2,400)
            sol = odeint(hybrid_swing, y0, t, args = (beta, R, m, M, g, d, d_tan, b, d_prime2))
            sol = np.array(sol)
            sol_abs = np.abs(sol)
            
            index = 0
                
            if k==0 or k ==7:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,1]>0:
                        if sol_abs[:,1].min(axis=0)>0.1 or (sol[l,0]-np.pi<0 and sol[l+1,0]-np.pi>0):
                            if sol[l,0]-np.pi<0 and sol[l+1,0]-np.pi>0:
                                diverging = True
                                index = l
                                previous_phi = -np.pi
                                previous_omega = sol[index,1]
                                break
                        else:
                            if sol[l,1]>0 and sol[l+1,1]<0:
                                index = l
                                previous_phi = sol[index, 0]
                                previous_omega = 0.0
                                break
                            
                    elif sol[1,1]<0:
                        if sol[l,0]-(np.pi/2)>0 and sol[l+1,0]-(np.pi/2)<0:
                            index = l
                            previous_phi = np.pi/2
                            previous_omega = sol[index,1]
                            break
            
                
            elif k==2 or k==6:
                    #index = np.where(sol_abs[:,1]==sol_abs[:,1].min(axis=0))
                    #previous_phi = sol[index,0]
                    #previous_omega = 0.0
                for o in range(0,len(sol)-1):
                    if sol[0,1]<0:
                        if sol[o,0]+(np.pi/2)>0 and sol[o+1,0]+(np.pi/2)<0:
                            index = o
                            previous_phi = -np.pi/2
                            break
                    elif sol[0,1]>0:
                        if sol[o,0]-(np.pi/2)<0 and sol[o+1,0]-(np.pi/2)>0:
                            index = o
                            previous_phi = np.pi/2
                            break
                previous_omega = sol[index, 1]
                
            elif k==1 or k==5:
                for l in range(0,len(sol)-1):
                    #if sol_abs[:,0].min(axis=0)<0.2:
                        #index = np.where(sol_abs[:,0]==sol_abs[:,0].min(axis=0))
                        #previous_phi = 0.0
                        #previous_omega = sol[index,1]
                    if sol[0,0]>0:
                        if sol[l,0]>0 and sol[l+1,0]<0:
                            index = l
                            break
                    elif sol[0,0]<0:
                        if sol[l,0]<0 and sol[l+1,0]>0:
                            index = l
                            break
                previous_phi = 0.0
                previous_omega = sol[index, 1]
            
            elif k==3 or k ==4:
                for p in range(0, len(sol)-1):
                    if sol[0,1]<0:
                        if sol_abs[:,1].min(axis=0)>0.1 or (sol[p,0]+np.pi>0 and sol[p+1,0]+np.pi<0):
                            if sol[p,0]+np.pi>0 and sol[p+1,0]+np.pi<0:
                                diverging = True
                                index = p
                                previous_phi = np.pi
                                previous_omega = sol[index,1]
                                break
                        else:
                            if sol[p,1]<0 and sol[p+1,1]>0:
                                index = p
                                previous_phi = sol[index, 0]
                                previous_omega = 0.0
                                break
                            
                    elif sol[1,1]>0:
                        if sol[p,0]+(np.pi/2)<0 and sol[p+1,0]+(np.pi/2)>0:
                            index = p
                            previous_phi = -np.pi/2
                            previous_omega = sol[index,1]
                            break
                            
                            
                
            if (j>=4*gradient):
                plt.plot(sol[0:index,0], sol[0:index,1], color=(grad_b[j-4*gradient],0,0))
            else:
                plt.plot(sol[0:index,0], sol[0:index,1], color=(1,grad_a[j],grad_a[j]))

            plt.xlabel('Phi(rad)')
            plt.ylabel('Phi_dot(rad/s)')
            #k=k+1
            
            all_phi = np.append(all_phi,sol[0:index,0])
            all_phidot = np.append(all_phidot,sol[0:index,1])
            all_time = np.append(all_time, t[0:index]+all_time[-1])
        
            if (j==n-1):
                t_p = t_p + t[index]
                
    plt.grid()        
    plt.show()

    plt.plot(all_time[1:len(all_time)-1], all_phi[1:len(all_phi)-1], color = 'g', label='phi(t)')   
    plt.plot(all_time[1:len(all_time)-1], all_phidot[1:len(all_phidot)-1], color = 'b', label = 'omega(t)')
    plt.xlabel('Time(s)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
    print (np.abs(all_phidot).max(axis=0))
    print (t_p)
    


all_phi = np.array([0])
all_phidot = np.array([0])
all_time = np.array([0])
hybrid(5*np.pi/8, 0, 0.9, 0.4, 0.5, 1.2, 9.8, 0.105, 0.15, 0.4, 0.5, 30, all_phi, all_phidot, all_time)
spinning_hybrid(-np.pi, 0.001, 0.9, 0.4, 0.5, 1.2, 9.8, 0.114, 0.15, 0.40, 0.50, 20, all_phi, all_phidot, all_time)


 

              