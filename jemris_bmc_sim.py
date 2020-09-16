# import modules

import numpy as np
import h5py
import os
import shutil as sh
import matplotlib.pyplot as plt
import multiprocessing, tempfile


# help functions

def gamma():
    return 42.576375 * 2 * np.pi       # [MHz/T]


def calc_flipangle(B1,tp): 
    ''' calculate flipangle from pulse amplitude and duration
    B1 in uT
    tp in s
    alpha in degrees '''

    return np.rad2deg(gamma() * B1 * tp * 10**(-3))


def calc_w0(B0):
    ''' calculate angular frequency from B0
    f0 = gamma_stern*B0 [MHz] 
    w0 [rad/ms]
    '''
    
    return gamma()*B0 * 10**(-3)

    
def calc_offset(w, w0):
    # w in ppm
    # dw in rad/ms

    return w * w0 


def add_dw2FREQ(FREQ,dw):
    if np.any(abs(FREQ) <1E-6):
        FREQ=FREQ
    else:
        FREQ= np.append(FREQ,0)
        
    if np.size(dw)==1:
        if np.any(FREQ==dw):
            FREQ=FREQ
        else:
            FREQ=np.append(FREQ,dw)
            FREQ=np.append(FREQ,-dw)

    else: 
        for i in range(0,np.size(dw)):
            if np.any(FREQ==dw[i]):
                FREQ=FREQ
            else:
                FREQ=np.append(FREQ,dw[i])
                FREQ=np.append(FREQ,-dw[i])
    FREQ=np.append(FREQ,100)
    return np.sort(FREQ)

# create mpsample

def mpsample(pools,water_T1,water_T2,B0):
    '''
    create multi pool sample with n-pools for Bloch McConnel Simulation
    '''
    
    fb =  np.array([p["fb"] for p in pools])
    dwb = np.array([p["dwb"] for p in pools])
    kb =  np.array([p["kb"] for p in pools])
    
    T1 = np.array([water_T1,]+[p["T1"] for p in pools])
    T2 = np.array([water_T2,]+[p["T2"] for p in pools])
    
    ### init sample matrix
    NP = 5             # of spin properties: M0, T1, T2, T2*, DB
    NC = 1+len(pools)  # of compartments (spin pools) (NC=1, only water)
    NX = 1             # of points on x
    NY = 1             # of points on y
    NZ = 2             # of points on z (at least 2 needed on z, otherwise not a MP-sample?)

    A=np.zeros([NP,NC,NX,NY,NZ])
    
    ### fill sample matrix
    # M0 of pools / proton fraction
    M0a = 100             # M0 of 1st pool (free)
    A[0,0,:,:,:] = M0a

    for i in range(0,np.size(fb)):
        A[0,i+1,:,:,:] = fb[i]*M0a  # M0 of n-th pool (bound)
    
    # relaxation rates
    # first pool (free)
    A[1,0,:,:,:] = 1/(T1[0])    # R1 = 1/T1
    A[2:4,0,:,:,:] = 1/(T2[0])  # R2 = R2*

    # bound pools 
    for i in range(1,np.size(T1)):
            A[1,i,:,:,:]      = 1/(T1[i])    # R1=1/T1
            A[2:4,i,:,:,:]    = 1/(T2[i])    # R2 = R2*

    # frequency offset
    w0 = calc_w0(B0)
    dw = calc_offset(dwb,w0)

    for i in range(0,np.size(dw)):
        A[4,i+1,:,:,:] = dw[i];   # off-res 2nd pool
            
            
    # exchange matrix
    E=np.zeros([NC,NC])  # only lower triangle affects results

    for i in range(0,np.size(kb)):
            E[i+1,0]=kb[i]*fb[i];        # unit: kHz/ ms^(-1)   

            
    ### write mpsample    
    
    if os.path.exists('mpsample.h5'): os.remove('mpsample.h5')
    
    hf = h5py.File('mpsample.h5', 'w')
    s = hf.create_group('sample')
    s.create_dataset('data', data=np.transpose(A))
    s.create_dataset('resolution',data=[1,1,1])
    s.create_dataset('offset', data=[0,0,0])
    hf.create_dataset('exchange', data=np.transpose(E))

    hf.close()        
    
    return
   
    
def start_cw_sim(arg):
    '''
    
    '''
 
    FREQ, td, alpha, path_ = arg

    with tempfile.TemporaryDirectory() as path:
        os.chdir(path)
    
        sh.copyfile(os.path.join(path_,'mpsample.h5'), 'mpsample.h5')
        sh.copyfile(os.path.join(path_,'simu_bmc.xml'), 'simu_bmc.xml')
        os.system('sed -e "s/XXX/' + str(-1*FREQ) + '/" -e "s/YYY/' + str(td) + '/" -e "s/ZZZ/'+ str(alpha)+ '/" ' + os.path.join(path_,'mt_rect.xml.form')+ ' > mt.xml')
        os.system('jemris simu_bmc.xml')
    
        S = h5py.File('signals.h5','r')
        signal=S.get('signal')
        channels = signal.get('channels')
        C = channels.get('00')
        Z=C[0,2]
        S.close()

    return Z


def run_cw_sim(td=5000, B1=5, frequency=np.array([-45,-30,0,30,45]), B0=7, n_p=None, verbose=True):
    '''
    '''
    
    ### define/calculate simulation parameters

    w0 = calc_w0(B0)
    FREQ = frequency*w0
    Z = np.zeros_like(FREQ)

    alpha = calc_flipangle(B1,td);
    path_ = os.getcwd()
    
    # execute simulation over all frequencies
    if n_p==None:
        n_p=np.max(np.array([1,(multiprocessing.cpu_count()//2)]))

    p = multiprocessing.Pool(processes=n_p)
    Z = p.map(start_cw_sim, [(i,td,alpha,path_) for i in FREQ])
 
    if verbose:
        plt.plot(frequency, Z/Z[0])
        plt.ylim([0,1.01])
        plt.xlabel('$dw [ppm]$',size=12)
        plt.ylabel('$M_{sat}/M_0$',size=12)
        plt.xlim([+frequency[-1],frequency[0]])
    
    return (Z, FREQ/w0)


def start_free_sim(arg):
     
    FREQ, sequence, path_ = arg

    with tempfile.TemporaryDirectory() as path:
        os.chdir(path)
    
        sh.copyfile(os.path.join(path_,'mpsample.h5'), 'mpsample.h5')
        sh.copyfile(os.path.join(path_,'simu_bmc.xml'), 'simu_bmc.xml')
        os.system('sed -e "s/XXX/' + str(-1*FREQ) + '/" ' + os.path.join(path_,sequence) + ' > mt.xml')
        os.system('jemris simu_bmc.xml')
    
        S = h5py.File('signals.h5','r')
        signal=S.get('signal')
        channels = signal.get('channels')
        C = channels.get('00')
        Z=C[0,2]
        S.close()

    return Z


def run_sim(sequence='mt_pulsed.xml', frequency=np.array([-45,-30,0,30,45]), B0=7, n_p=None, verbose=True):
    '''
    '''
    
    ### define/calculate simulation parameters
    
    w0 = calc_w0(B0)
    FREQ = frequency*w0
    Z = np.zeros_like(FREQ)
    path_ = os.getcwd()

    # execute simulation over all frequencies
    if n_p==None:
        n_p=np.max(np.array([1,(multiprocessing.cpu_count()//2)]))

    p = multiprocessing.Pool(processes=n_p)
    Z = p.map(start_free_sim, [(i,sequence,path_) for i in FREQ])

    if verbose:
        plt.plot(frequency, Z/Z[0])
        plt.ylim([0,1.01])
        plt.xlabel('$dw [ppm]$',size=12)
        plt.ylabel('$M_{sat}/M_0$',size=12)
        plt.xlim([frequency[-1],frequency[0]])

    return (Z, FREQ/w0)