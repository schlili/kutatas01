import numpy as np
from scipy.integrate import *
from numpy.fft import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib.ticker import ScalarFormatter
from types import SimpleNamespace
import pickle

class Modell:
    '''
    ---Az osztály tagjai a különböző paraméterekkel
    kapott adatsorok.
    ---Létrehozáshoz egy fájlnevet adunk meg:
       ez tartalmazza a paramétereket.
    '''
    
    def __init__(self,file=None):
        
        if file==None:
            self.d=0
            self.m=0
            self.R=0
            self.rho=0
            self.k=0
            self.o=0
            self.b=0
            self.u=np.array([])
            self.t=np.array([])
            self.n=0
            
        else:
            with open(file, 'rb') as f:
                params = pickle.load(f)
                
            self.d=params.d
            self.m=params.m
            self.R=params.R
            self.rho=params.rho
            self.k=params.k
            self.o=params.o
            self.b=params.b
            self.u=params.u
            self.t=params.t
            self.n=params.n

    def details(self):
        print('d: ' ,self.d)
        print('m: ' ,self.m)
        print('R: ' ,self.R)
        print('rho: ' ,self.rho)
        print('k: ' ,self.k)
        print('o: ' ,self.o)
        print('b: ' ,self.b)
        
    def change_params(self,params):
        'Paraméterek SimpleNamespace objektumban.'
        
        self.d=params.d
        self.m=params.m
        self.R=params.R
        self.rho=params.rho
        self.k=params.k
        self.o=params.o
        self.u=params.u       
        self.b=params.b
        self.t=params.t

        
    def start_maker(self,n):
    
        self.n=n

        phi_0=np.linspace(0,2*np.pi,self.n, endpoint=False)
        omega_0=np.zeros(self.n)
        theta=np.array([0])
        Omega=np.array([self.o])

        self.X0=np.concatenate((phi_0,omega_0,theta, Omega))
    
    def diffeq(self, t, X):
        """
        A modell: R sugarú körön egyenletesen elrendezett golyók összekötve valamilyen rugóval.
        Körülöttük egy másik golyó halad.

        """

        N=self.n
        d,m,R,rho,k,b=(self.d, self.m, self.R, self.rho, self.k, self.b)  #bejövő paraméterek
        l=2*R*np.sin(np.pi/(N)) #rugók nyugalmi hossza

        phi = np.array(X[:N]) #első N darab elem a szögek
        omega = np.array(X[N:2*N])  #második N darab a szögsebességek
        theta = np.array([X[-2]]) #utolsó előtti elem a kezdeti szög
        Omega = np.array([X[-1]]) #utolsó a szögsebessége

        phi_m1 = np.roll(phi,-1)
        phi_p1 = np.roll(phi,1)

          #belső golyók szöggyorsulása:
        beta = (d/(m*R)) * ( R*np.sin(phi_m1-phi) -
                               R*np.sin(phi-phi_p1) +
                              (np.sqrt(2)*l*np.sin(phi - phi_p1))/(2*np.sqrt(1-np.cos(phi-phi_p1))) -
                              (np.sqrt(2)*l*np.sin(phi_m1 - phi))/(2*np.sqrt(1-np.cos(phi_m1-phi))) +
                               (rho*k*b/(d))*((np.exp( -b*np.sqrt(R**2 - 2*R*rho*np.cos(theta - phi) + rho**2))*np.sin(theta - phi)) /
                              (np.sqrt(R**2 - 2*R*rho*np.cos(theta - phi) + rho**2 )))
                             )


          #külső golyó szöggyorsulása:
        Beta = np.array([((-R*k*b)/(m*rho)) * np.sum(np.exp( -b*np.sqrt(R**2 - 2*R*rho*np.cos(theta - phi) + rho**2))*np.sin(theta - phi)
                                          / np.sqrt(R**2 - 2*R*rho*np.cos(theta - phi) + rho**2 ))])

        U = np.concatenate((omega, beta, Omega, Beta))

        return U
        
    def solve_ivp(self,t, rtol=1e-3, atol=1e-6, max_step=0.01): #ez jó
        
        self.atol=atol
        self.rtol=rtol
        t_eval = np.linspace(0, t, t*100) #időpontok, ahol kiértékel
        solution=solve_ivp(self.diffeq, (0,t), self.X0, method='RK45', t_eval=t_eval, rtol=rtol, atol=atol, max_step=max_step)
        self.u=(solution.y).T
        self.t=(solution.t)
    
    def solution_divider(self):

        self.phi=self.u[:,:self.n]
        self.omega=self.u[:,self.n:(2*self.n)]
        self.theta=self.u[:,-2]
        self.Omega=self.u[:,-1]

        
    def save(self,file):
        params=SimpleNamespace(d=self.d, m=self.m, R=self.R, rho=self.rho, k=self.k, o=self.o, u=self.u, t=self.t, n=self.n, b=self.b)

        with open(file +'.pkl', 'wb') as f:
            pickle.dump(params, f)

            
    def phonon_spectrum(self, ax):

        v_belso=(self.R*self.omega)
        t=self.t
        kran_b = np.linspace(-np.pi, np.pi, self.n)
        wran_b = np.linspace(-np.pi/np.diff(t)[0],np.pi/np.diff(t)[0], len(t))
        im = ax.pcolormesh(kran_b, wran_b, np.abs(np.fft.fftshift(np.fft.ifft(np.fft.fft(v_belso, axis=1),axis=0), axes=(1,0)))**2, norm=LogNorm())

        ax.set_ylim(0,3)
        ax.get_figure().colorbar(im, ax=ax)
        ax.set_ylabel(r'$\omega$ körfrekvencia', size=13)
        ax.set_xlabel(r'$k$ hullámszám', size=13)
        ax.set_title(r'$\omega (k)$', size=14) 
        
    def energy_calc(self):

        N=self.n
        T=self.t
        d,m,R,rho,k,b=(self.d, self.m, self.R, self.rho, self.k, self.b) 
        l=2*R*np.sin(np.pi/(N))    

        phi_p1 = np.roll(self.phi,1, axis=1)

#         E=[]
#         K_in=[]
#         K_out=[] 
#         V_in=[]
#         V_out=[]
        
        V_in=1/2 * d * (R*np.sqrt(2)* np.sqrt(1-np.cos(self.phi-phi_p1)) - l)**2
        V_out=-k*np.exp(-1*b*np.sqrt(R**2-2*R*rho*np.cos(self.theta[:, np.newaxis]-self.phi)+rho**2))
        K_in=1/2 * m * R**2 * self.omega**2
        K_out=1/2 * m * self.Omega**2 *rho**2
        
        E=np.sum(V_in+V_out+K_in, axis=1)+K_out

#         for t in range(len(T)):
#             V_in_temp= 1/2 * d * (-1*R*np.sqrt(2)* np.sqrt(1-np.cos(self.phi[t,:]-phi_p1[t,:])) + l)**2
#             V_out_temp= -k*np.exp(-1*b*np.sqrt(R**2-2*R*rho*np.cos(self.theta[t]-self.phi[t,:])+rho**2))
#             K_in_temp=1/2 * m * R**2 * self.omega[t,:]**2
#             K_out_temp=1/2 * m * self.Omega[t]**2 *rho**2
            
#             E_temp=np.sum(V_in_temp+V_out_temp+K_in_temp)+K_out_temp

#             E.append(E_temp)

#             K_in.append(K_in_temp)
#             K_out.append(K_out_temp)
#             V_in.append(V_in_temp)
#             V_out.append(V_out_temp)

        self.E=np.array(E)
        self.K_in=np.array(K_in)
        self.K_out=np.array(K_out)
        self.V_out=np.array(V_out)
        self.V_in=np.array(V_in)
        
#     def energy_plotter(self):
        
#         fig,ax=plt.subplots(figsize=(16,4))
        
#         ax.set_title('Összenergia', size=15)
#         ax.set_ylabel(r'$E_{ossz}$', size=13)
#         ax.set_xlabel(r'$t$', size=13)
#         ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#         ax.plot(self.t,self.E, marker='x', linestyle=':', label=f'atol= {self.atol} ; rtol={self.rtol}')
#         ax.legend()