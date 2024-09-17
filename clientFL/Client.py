import torch
from torch.utils.data import DataLoader
import copy
from torch.nn import functional as F
import time
import random
import math
class Client(object):
    
    def __init__(self, local_dataset, local_model,loss, learning_rate, momentum,B,N0,PK,frequency,CpuCycle,X,bandwidth,gk,cores,battery):
        """
        Initializes a client for FL.

        Parameters:
        - client_name (str): Name of the client.
        - local_dataset: The local dataset for the client.
        - local_model: The local model for the client.
        - numbercores (int): Number of cores available on the client.
        - frequency: Distribution of frequency.
        - bandwidth: Distribution of bandwidth.
        - loss: Loss function used for training.
        - B (int): Size of a batch.
        - learning_rat (float) e: Learning rate for the optimizer.
        - momentum (float): Momentum for the optimizer.
        """
        
        
        # name of the client
        # number of cores
        self.E=10
        self.frequency = frequency      
        self.f=random.choice(self.frequency)
        self.B=B
        self.bandwidth =  bandwidth #带宽
        self.N0=N0 #
        self.PK=PK
        self.CpuCycle=CpuCycle
        self.X=X
        self.gk=gk
        self.local_data = DataLoader(local_dataset, batch_size= B, shuffle=True)
        self.D=len(self.local_data)
        self.local_model = copy.deepcopy(local_model)
        self.loss_func = loss
        self.numbercores=cores
        self.count=0
        self.total_energy=0
        # the optimizer
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr= learning_rate , momentum = momentum)
        self.battery=battery
        
    def set_weights(self, global_parameters):
        """
        Set the weights of the local model using the global model's parameters.

        Parameters:
        - global_parameters: The global model parameters recieved from the server.
        """
        self.local_model.load_state_dict(global_parameters)

         
    def train(self, global_parameters, E, mu, verbos=0):
        """
        Train the local model for a task other than fall detection.

        Parameters:
        - global_parameters: The global model parameters.
        - E (int): Local number of epoch.
        - mu: FedProx parameter.
        - verbos (int): Verbosity level for printing training information.

        Returns:
        - A deep copy of the state dictionary of the trained local model.
        """
        self.f=random.choice(self.frequency)
        self.E=E
        # Initialize local model parameters by global_parameters
        self.local_model.load_state_dict(global_parameters)
        # Start local training
        self.local_model.train()
        for iter in range(E):
            if (verbos == 1) :
                print("Client : ",self.client_name," Iteration :",iter+1)
            index=0
            for images, labels in self.local_data:
                index+=1
                # Initialize the gradients to 0
                self.local_model.zero_grad()
                # Probability calculation for batch i images
                log_probs = self.local_model(images)
                # Loss calculation
                loss = self.loss_func(log_probs, labels)
                
                # The addition of the term proximal
                if (mu != 0 and iter > 0):
                        for w, w_t in zip(self.local_model.parameters(), global_parameters.values()):
                            loss += mu / 2. * torch.pow(torch.norm(w.data - w_t.data), 2)
                
                # Calculation of gradients
                loss.backward()
                
                # Update of the parameters
                self.optimizer.step()
                
        return copy.deepcopy(self.local_model.state_dict())


    def get_size(self):
        """
        Get the size of the local dataset.

        Returns:
        - The size of the local dataset.
        """
        return len(self.local_data.dataset)

    def get_model(self):
        """
        Get the local model.

        Returns:
        - The local model.
        """
        return self.local_model

    def delete_model(self):
        """
        Delete the local model to free up resources.
        """
        del self.local_model

    def calculate_loss(self):
        """
        Calculate the loss of the current model on the local dataset.

        Returns:
        - The calculated loss.
        """
        self.local_model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.local_data:
                output = self.local_model(data)
                loss = self.loss_func(output, target)
                total_loss += loss.item() * len(data)
                total_samples += len(data)

        # Return average loss over all samples
        return total_loss / total_samples
    
    def get_state(self):
        """
        Get the state of the client, including model parameters, dataset size, computing resources, etc.
        
        Returns:
        - A list representing the flattened state, ready for processing by flatten_state method.
        """
        # 展平模型参数并转换为列表形式
        model_parameters = [param.flatten().tolist() for param in self.local_model.parameters()]
        # 展开模型参数的嵌套列表
        model_parameters = [item for sublist in model_parameters for item in sublist]

        # 其他状态信息
        dataset_size = [self.get_size()]  # 转换为列表
        numbercores = [self.numbercores]  # 转换为列表
        frequency = [random.choice(self.frequency)]  # 转换为列表
        bandwidth = [self.bandwidth]  # 转换为列表

        # 返回合并后的状态信息列表
        return model_parameters + dataset_size + numbercores + frequency + bandwidth

    def CalculationTrainEngry(self,):
        return ((self.X*self.CpuCycle*self.D*self.f*self.f*self.E*self.numbercores))/self.battery
    def CalculationTrainTime(self,):
        return (self.E*self.CpuCycle*self.D)/(self.numbercores*self.f)
    def CalculationTranslationTime(self,bandwidth):
        return  (3588096/(bandwidth))
    def CalculationTranslationEngry(self,bandwidth):
        return  (self.PK*self.CalculationTranslationTime(bandwidth))/self.battery
    def EnergyFunction(self,bandwidth):
        return math.pow(0.1,self.count)/(self.CalculationTrainEngry()+self.CalculationTranslationEngry(bandwidth))