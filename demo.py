import numpy as np
import matplotlib
matplotlib.use("module://matplotlib_pyodide.html5_canvas_backend")
from matplotlib import pyplot as plt
hbar = 6.582E-16  ## in eV.s
Me = 510.998E3  ## eV/c2
speedoflight = 299792458  ## m/s
params_3R_MoSSe_SS_VBM_Gamma = dict({'name':'3R-MoSSe,SS,VBM,Gamma',
                   'W1':6.1, 'W2':0.2,'W3':0.2, 'phi_W1':0,'phi_W2':0, 'phi_W3':0,
                   'V1':7.8, 'V2':0.2,'V3':0.3, 'phi_V1':0,'phi_V2':0, 'phi_V3':0,
                   'm0':-0.4, 'U1':0.01,'phi_U1':0,'alpha_R':0,'lattice_constant':3.25})
params_3R_WSTe_SS_VBM_Gamma = dict({'name':'3R-WSTe,SS,VBM,Gamma',
                   'W1':9.6, 'W2':2.8,'W3':2.0, 'phi_W1':0,'phi_W2':180, 'phi_W3':0,
                   'V1':14.7, 'V2':33.6,'V3':13.4, 'phi_V1':180,'phi_V2':0, 'phi_V3':180,
                   'm0':-1.3, 'U1':0.09,'phi_U1':0,'alpha_R':0,'lattice_constant':3.25})
params_3R_WSSe_SeSe_VBM_Gamma = dict({'name':'3R-WSSe,SeSe,VBM,Gamma',
                   'W1':7.6, 'W2':1.3,'W3':0.6, 'phi_W1':0,'phi_W2':-180, 'phi_W3':-180,
                   'V1':8.9, 'V2':2.8,'V3':0.5, 'phi_V1':180,'phi_V2':180, 'phi_V3':0,
                   'm0':-0.7, 'U1':0.02,'phi_U1':0,'alpha_R':0,'lattice_constant':3.25})
params_2H_WSSe_SeSe_VBM_Gamma = dict({'name':'2H-WSSe,SeSe,VBM,Gamma',
                   'W1':7.2, 'W2':0.1,'W3':0.7, 'phi_W1':148.1,'phi_W2':-109.9, 'phi_W3':-60.5,
                   'V1':3.6, 'V2':1.2,'V3':1.5, 'phi_V1':93.4,'phi_V2':178.9, 'phi_V3':29.7,
                   'm0':-0.72, 'U1':0.03,'phi_U1':125,'alpha_R':0,'lattice_constant':3.25})
def moire_reciprocal_vectors(a=3.25, theta_deg=4.0):
    b = (2*np.pi/a) * np.sqrt(4/3);  ## primitive lattice and reciprocal: triangular lattice b != 2pi/a
    theta_rad = theta_deg * (np.pi/180);  ## converting degrees to radians
    L = a/(2*np.sin(theta_rad/2)) ## moire length scale
    g = (2*np.pi/L) * np.sqrt(4/3) ## moire reciprocal lattice const. (non-orthogonal basis!)
#### Older version of the g-vectors. The 2nd version matches the PRB.
#     g11 = g*np.array([1,0])
#     g12 = g*np.array([1/2,np.sqrt(3)/2])
#     g13 = g*np.array([-1/2,np.sqrt(3)/2])
    g11 = g*np.array([np.sqrt(3)/2,-1/2])
    g12 = g*np.array([np.sqrt(3)/2,1/2])
    g13 = g*np.array([0,1])
    g14 = -g11; g15 = -g12; g16 = -g13;
    g21 = g11+g12; g22 = g12+g13; g23 = g13+g14;
    g24 = g14+g15; g25 = g15+g16; g26 = g16+g11;
    g31 = 2*g11; g32 = 2*g12; g33 = 2*g13;
    g34 = 2*g14; g35 = 2*g15; g36 = 2*g16
    return np.array([0*g11, g11,g12,g13,g14,g15,g16,g21,g22,g23,g24,g25,g26,g31,g32,g33,g34,g35,g36])
## Uses "params" values to define the hamiltonian
def Hk(k,params,glist):
    m0 = params['m0']
    V = np.array([params['V1']*np.exp(1j*np.pi*params['phi_V1']),params['V2']*np.exp(1j*np.pi*params['phi_V2']),params['V3']*np.exp(1j*np.pi*params['phi_V3'])])
    g0,g11,g12,g13,g14,g15,g16,g21,g22,g23,g24,g25,g26,g31,g32,g33,g34,g35,g36 = glist
    Hk = np.zeros((19,19),dtype=complex)
    ke_coeff = 1000*hbar**2/(2*Me/speedoflight**2) * m0 /(1E-10)**2 ## meV A^2; sign of m0 determines CB vs VB (+,- respectively)
    Hk[0,0:7] = np.array([ke_coeff*np.dot(k,k), V[0].conj(),V[0],V[0].conj(),V[0],V[0].conj(),V[0]])
    Hk[0,7:19] = np.array([V[1].conj(),V[1],V[1].conj(),V[1],V[1].conj(),V[1],V[2].conj(),V[2],V[2].conj(),V[2],V[2].conj(),V[2]])
    Hk[1,0:7] = np.array([V[0],ke_coeff*np.dot(k+g11,k+g11), V[0].conj(),V[1].conj(),V[2],V[1],V[0].conj()])
    Hk[1,7:19] = np.array([V[0],V[2].conj(),0,0,V[2].conj(),V[0],V[0].conj(),V[1],0,0,0,V[1].conj()])
    Hk[2,0:7] = np.array([V[0].conj(),V[0],ke_coeff*np.dot(k+g12,k+g12), V[0],V[1],V[2].conj(),V[1].conj()])
    Hk[2,7:19] = np.array([V[0].conj(),V[0].conj(),V[2],0,0,V[2],V[1],V[0],V[1].conj(),0,0,0])
    Hk[3,0:7] = np.array([V[0],V[1],V[0].conj(),ke_coeff*np.dot(k+g13,k+g13), V[0].conj(),V[1].conj(),V[2]])
    Hk[3,7:19] = np.array([V[2].conj(),V[0],V[0],V[2].conj(),0,0,0,V[1].conj(),V[0].conj(),V[1],0,0])
    Hk[4,0:7] = np.array([V[0].conj(),V[2].conj(),V[1].conj(),V[0],ke_coeff*np.dot(k+g14,k+g14), V[0],V[1]])
    Hk[4,7:19] = np.array([0,V[2],V[0].conj(),V[0].conj(),V[2],0,0,0,V[1],V[0],V[1].conj(),0])
    Hk[5,0:7] = np.array([V[0],V[1].conj(),V[2],V[1],V[0].conj(),ke_coeff*np.dot(k+g15,k+g15), V[0].conj()])
    Hk[5,7:19] = np.array([0,0,V[2].conj(),V[0],V[0],V[2].conj(),0,0,0,V[1].conj(),V[0].conj(),V[1]])
    Hk[6,0:7] = np.array([V[0].conj(),V[0],V[1],V[2].conj(),V[1].conj(),V[0],ke_coeff*np.dot(k+g16,k+g16)])
    Hk[6,7:19] = np.array([V[2],0,0,V[2],V[0].conj(),V[0].conj(),V[1].conj(),0,0,0,V[1],V[0]])
    Hk[7:19,0:7] = Hk[0:7,7:19].T.conj()
    Hk[7,8] = Hk[7,12] = Hk[9,10] = Hk[11,12] = V[1].conj()
    Hk[8,9] = Hk[10,11] = V[1];
    Hk[7,13] = Hk[9,15] = Hk[11,17] = V[0];
    Hk[7,14] = Hk[9,16] = Hk[11,18] = V[0].conj();
    Hk[8,14] = Hk[10,16] = Hk[12,18] = V[0].conj();
    Hk[8,15] = Hk[10,17] = Hk[12,13] = V[0];
    Hk[13,14] = Hk[13,18] = Hk[15,16] = Hk[17,18] = V[2].conj();
    Hk[14,15] = Hk[16,17] = V[2]
    Hk[7:19,7:19] += Hk[7:19,7:19].T.conj()  ## make hermitian
    Hk[7,7] = ke_coeff*np.dot(k+g21,k+g21); Hk[8,8] = ke_coeff*np.dot(k+g22,k+g22); Hk[9,9] = ke_coeff*np.dot(k+g23,k+g23);
    Hk[10,10] = ke_coeff*np.dot(k+g24,k+g24); Hk[11,11] = ke_coeff*np.dot(k+g25,k+g25); Hk[12,12] = ke_coeff*np.dot(k+g26,k+g26);
    Hk[13,13] = ke_coeff*np.dot(k+g31,k+g31); Hk[14,14] = ke_coeff*np.dot(k+g32,k+g32); Hk[15,15] = ke_coeff*np.dot(k+g33,k+g33);
    Hk[16,16] = ke_coeff*np.dot(k+g34,k+g34); Hk[17,17] = ke_coeff*np.dot(k+g35,k+g35); Hk[18,18] = ke_coeff*np.dot(k+g36,k+g36);
    return Hk
## Uses "params" values to define the moire potential
## requires "glist" of moire G-vectors
def delta(x,y,params,glist):
    r =np.stack([x,y],axis=-1)
    V = np.array([params['V1'],params['V2'],params['V3']])
    phi_V = np.array([params['phi_V1'],params['phi_V2'],params['phi_V3']])*np.pi/180 ## convert to radians
    delta = 0+0j
    for l in range(3):
        for jj in range(1,7):
            delta += V[l]*np.exp(((-1)**(jj+1))*1j*phi_V[l])*np.exp(-1j*np.matmul(r, glist[jj+6*l]))
    return np.real_if_close(delta)
## Uses "params" values to define the magnitude-squared of the wavefunction
## requires "glist" of moire G-vectors
def psi2(x,y,evect,glist):
    r =np.stack([x,y],axis=-1)
    psi = 0+0j
    for l in range(19):
        psi += evect[l]*np.exp(-1j*np.matmul(r, glist[l]))
    return np.real_if_close(np.conjugate(psi)*psi)
def bz_distance(lengths, num_per_section=20):
    #kdist = np.arange(0.0, lengths[0], lengths[0]/(num_per_section-1))
    dist=0.0
    kdist = []
    for k in lengths:
        kdist = np.append(kdist, np.linspace(dist, dist+k, num_per_section, endpoint=False))
        dist = dist + k
    kdist = np.append(kdist, [dist])
    return kdist
## Depends on the lattice constant and the twist angle.
## The endpoints of the path could be changed... that would be fancier and not needed on a first pass.
def make_kpath(a=3.25, theta_deg=4.0, lengths=[1.0,np.sqrt(3)/2,1/2], num_per_section=20):
    kdist = bz_distance(lengths,num_per_section=num_per_section)
    kpath = np.zeros((3*num_per_section+1,2))
    glist = moire_reciprocal_vectors(a, theta_deg)
    for i, x in enumerate(np.linspace(1,0,num_per_section,endpoint=False)):
        kpath[i] = x*(glist[1]+glist[2])/3
    for i, x in enumerate(np.linspace(0,1,num_per_section,endpoint=False)):
        kpath[i+num_per_section] = x*(glist[1])/2
    for i, x in enumerate(np.linspace(0,1,num_per_section,endpoint=False)):
        kpath[i+num_per_section*2] = (glist[1])/2 + x*((glist[1]+glist[2])/3-glist[1]/2)
    kpath[-1] = (glist[1]+glist[2])/3
    return kpath, kdist, glist
kpath, kdist, glist = make_kpath(a=3.25, theta_deg=4.0)
def make_plot(parameters, theta_deg=4.0):
    a = parameters['lattice_constant']
    theta_rad = np.pi*theta_deg/180.
    L = a/(2*np.sin(theta_rad/2)) ## moire length scale
    kpath, kdist, glist = make_kpath(a=a, theta_deg=theta_deg)
    X, Y = np.meshgrid(np.linspace(-L,L,101),np.linspace(-np.sqrt(3)*L/2,np.sqrt(3)*L/2,101))
    ############
    plt.figure(figsize=(24,7))
    plt.subplot(131)
    Z = delta(X,Y,parameters,glist)
    plt.contourf(X,Y,Z,levels=23,cmap='jet')
    plt.colorbar(shrink=0.7)
    plt.gca().set_aspect('equal')
    ############
    plt.subplot(132)
    evalarray = np.zeros((61,19))
    for i, k in enumerate(kpath):
        evalarray[i] = np.linalg.eigh(Hk(k,parameters,glist))[0]
    for bnd in range(19):
        plt.plot(kdist,evalarray[:,bnd])
    plt.xlim(kdist[0],kdist[-1])
    #plt.ylim(ymin,ymax)
    plt.xticks([kdist[0],kdist[20],kdist[40],kdist[-1]],['K','$\Gamma$','M','K']);
    plt.title(parameters['name']);
    plt.scatter(kdist[20],evalarray[20,-1],marker='*',color='y',edgecolor='k',s=300,zorder=2.5)
    ############
    plt.subplot(133)
    evals, evects = np.linalg.eigh(Hk(kpath[20],parameters,glist)) ## evectors are in columns, i.e. nth evector = evects[:,n]
    Z = psi2(X,Y,evects[:,-1],glist)
    plt.contourf(X,Y,Z,levels=23,cmap='viridis')
    plt.colorbar(shrink=0.7)
    plt.gca().set_aspect('equal')
    plt.show()
print("debug")
parameters = dict(params_3R_MoSSe_SS_VBM_Gamma)
make_plot(parameters, theta_deg=4.0)
parameters = dict(params_3R_WSTe_SS_VBM_Gamma)
make_plot(parameters, theta_deg=3.0)
parameters = dict(params_3R_WSSe_SeSe_VBM_Gamma)
make_plot(parameters, theta_deg=1.5)
parameters = dict(params_2H_WSSe_SeSe_VBM_Gamma)
make_plot(parameters, theta_deg=2.0)
