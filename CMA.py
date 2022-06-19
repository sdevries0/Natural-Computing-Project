# The code from https://arxiv.org/pdf/1604.00772.pdf was used and translated to work in Python.

import numpy as np
import network
import warnings
warnings.filterwarnings("ignore")

class Args:
    def __init__(self, N, lmbd):
        # Number of parameters
        self.N = N 
        # Initial mean of population
        self.xmean = np.random.rand(self.N) 
        # Step size
        self.sigma = 0.5 

        # Population size
        self.lmbd = lmbd 
        # Number of solutions to use for recombination
        self.mu = self.lmbd/2 
        # Initialise weights
        self.weights = (np.log(self.mu+1/2)-np.log(range(1,int(self.mu+1))).conj().T).reshape(-1,1)
        
        self.mu = int(np.floor(self.mu)) 
        # Normalize weights
        self.weights = self.weights/np.sum(self.weights) 
        # Variance-effective size of mu
        self.mueff=np.sum(self.weights)**2/np.sum(self.weights**2) 
        self.weights_diag = np.zeros((self.weights.shape[0],self.weights.shape[0]))
        np.fill_diagonal(self.weights_diag,self.weights)

        # Initialize learning rates and other hyperparameters
         # Time constant for cumulation for C
        self.cc = (4+self.mueff/self.N) / (self.N+4 + 2*self.mueff/self.N)
         # Time onstant for cumulation for sigma control
        self.cs = (self.mueff+2)/(self.N+self.mueff+5)
        # Learning rate for rank-one update of C
        self.c1 = 2 / ((self.N+1.3)**2+self.mueff) 
         # Learning rate for rank-mu update
        self.cmu = 2 * (self.mueff-2+1/self.mueff) / ((self.N+2)**2+2*self.mueff/2)
        # Damping for sigma
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.N+1))-1) + self.cs 

        # Initialize evolution paths
        self.pc = np.zeros((self.N)) 
        self.ps = np.zeros((self.N)) 
        # Eigenvectors
        self.B = np.eye(self.N) 
        # Eigenvalues
        self.D = np.eye(self.N) 
        self.D_Matrix = self.D
        # Covariance matrix
        self.C = self.B@self.D@(self.B@self.D).conj().T 
        self.eigeneval = 0 
        # Expectation of N(0,1)
        self.chiN=self.N**0.5*(1-1/(4*self.N)+1/(21*self.N**2))
        # Create arrays to store samples
        self.arz = np.zeros((self.N, int(self.lmbd)))
        self.arx = np.zeros((self.N, int(self.lmbd)))


class Agent():
    def __init__(self, env, network_hiddens):
        self.environment = env
        self.network_hiddens = network_hiddens
        self.network_input = self.environment.input_neurons
        self.network_output = self.environment.output_neurons

        # network_weights = (network_input*network_h1 + network_h1*network_h2 + network_h2*network_output)
        self.network_weights = self.network_input * self.network_hiddens[0] + np.sum([self.network_hiddens[i] * self.network_hiddens[i+1] for i in range(len(self.network_hiddens)-1)]) + self.network_hiddens[-1] * self.network_output
        self.network_biasses = np.sum(self.network_hiddens) + self.network_output
        self.network_params = self.network_weights + self.network_biasses

        # Calculate number of candidate solutions
        self.lmbd = 4+np.floor(3*np.log(self.network_params))

        # Initialize args
        self.weights_args = Args(self.network_weights, self.lmbd)
        self.bias_args = Args(self.network_biasses, self.lmbd)

        self.counteval = 0
        self.arfitness = np.zeros(int(self.lmbd))
        self.agents = []

        # Create agents
        for k in range(0, int(self.lmbd)):
            self.agents.append(network.Network())

        # Initialize the networks
        for agent in self.agents:
            agent.create_network(self.network_input,self.network_hiddens, self.network_output, 'softmax')

    def run_generation(self):
        for k in range(0,int(self.lmbd)):
            # Sample from N(0,I)
            self.weights_args.arz[:,k] = np.random.randn(self.weights_args.N)
            self.bias_args.arz[:,k] = np.random.randn(self.bias_args.N)

            # Compute samples from N(mean, CI)
            self.weights_args.arx[:,k] = self.weights_args.xmean + self.weights_args.sigma * (self.weights_args.B@self.weights_args.D @ self.weights_args.arz[:,k])
            self.bias_args.arx[:,k] = self.bias_args.xmean + self.bias_args.sigma * (self.bias_args.B@self.bias_args.D @ self.bias_args.arz[:,k])

            # Set weights and biasses in network
            self.agents[k].set_weights(self.weights_args.arx[:,k], self.bias_args.arx[:,k])
            
            # Compute fitness of agent
            self.arfitness[k] = self.environment.fitness(self.agents[k])

            self.counteval += 1

        # Sort fitnesses
        self.arindex = np.flip(np.argsort(self.arfitness))
        self.arfitness = self.arfitness[self.arindex]

        # Update weights parameters
        self.update(self.weights_args, self.arindex, self.arfitness, self.counteval)
        # Update bias parameters
        self.update(self.bias_args, self.arindex, self.arfitness, self.counteval)

        return self.arfitness[0]

    def update(self, args, arindex, arfitness, counteval):
        # Sort by fitness and update mean of x and z
        best_z = args.arz[:,arindex[0:args.mu]]
        args.xmean = np.dot(args.arx[:,arindex[0:args.mu]],args.weights)[:,0]
        args.zmean = np.dot(best_z,args.weights)

        # Update evolution paths
        args.ps = (1-args.cs)*args.ps + (np.sqrt(args.cs*(2-args.cs)*args.mueff)) * (args.B @ args.zmean)[:,0]
        args.hsig = int(np.linalg.norm(args.ps)/np.sqrt(1-(1-args.cs)**(2*counteval/args.lmbd))/args.chiN < 1.4+2/(args.N+1))
        args.pc = (1-args.cc)*args.pc + args.hsig * np.sqrt(args.cc*(2-args.cc)*args.mueff) * (args.B@args.D@args.zmean)[:,0]

        # Update covariance matrix C
        args.C = (1-args.c1-args.cmu) * args.C  + args.c1 * (args.pc.reshape(-1,1)@args.pc.reshape(-1,1).conj().T  + (1-args.hsig) * args.cc*(2-args.cc) * args.C)  + args.cmu  * (args.B@args.D@best_z) @ args.weights_diag @ (args.B@args.D@best_z).conj().T

        # Update step-size sigma
        args.sigma = args.sigma * np.exp((args.cs/args.damps)*(np.linalg.norm(args.ps)/args.chiN - 1)) 

        # Update B and D from C
        if (counteval - args.eigeneval) > (args.lmbd/(args.c1+args.cmu)/args.N/10):
            args.eigeneval = counteval
            args.C=np.triu(args.C)+np.triu(args.C,1).conj().T

            # Compute eigenvalues and eigenvectors
            [D_prime,args.B] = np.linalg.eig(args.C)
            args.B = np.flip(-1*args.B,axis=1)
            args.D = np.flip(D_prime*args.D_Matrix)
            args.D = np.diag(np.sqrt(np.diag(args.D)))

        # Check for flat fitnesses
        if arfitness[0] == arfitness[int(np.ceil(0.7*args.lmbd))]:
            args.sigma = args.sigma * np.exp(0.2+args.cs/args.damps)

