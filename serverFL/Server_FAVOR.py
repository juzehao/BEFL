import copy
import clientFL.Client as Client
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import torch.nn as nn
import RL.DQL as DQL
from sklearn.metrics import f1_score,recall_score,precision_score
from torch.nn import functional as F
import timeit

class Server_FAVOR(object):
    
    def __init__(self, num_clients, global_model, dict_clients, loss_fct, B, dataset_test, learning_rate, momentum, clients_info,bandwidth):
        """
        Initialize the Server_FLASHRL object.

        Parameters:
        - num_clients: The number of clients in the system.
        - global_model: The global model.
        - dict_clients: A dictionary containing information about each client.
        - loss_fct: The loss function used for training.
        - B: The size of the batch.
        - dataset_test: The test dataset used for evaluation.
        - learning_rate: The learning rate for training.
        - momentum: The momentum parameter for training.
        - clients_info: Information about the clients for simulation purposes.
        """
        self.N = num_clients
        self.bandwidth=bandwidth
        self.model = global_model 
        self.list_clients = [] 
        self.B = B
        self.dataset_test = dataset_test
        self.testdataloader = DataLoader(self.dataset_test, batch_size= self.B)
        self.dict_clients = dict_clients
        self.loss_function = copy.deepcopy(loss_fct)
        self.number_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.clients_info = clients_info
        self.create_clients(learning_rate, momentum)

   
    def create_clients(self, learning_rate, momentum):
        cpt = 0
        for client_name in self.dict_clients.keys():
            client = Client.Client(
                self.dict_clients[client_name],  # client information
                copy.deepcopy(self.model),  # copy of the global model
                copy.deepcopy(self.loss_function),  # copy of the loss function
                learning_rate,  # learning rate
                momentum,  # momentum
                self.B,
                self.clients_info[cpt][0],  # number of cores
                self.clients_info[cpt][1],  # frequency distribution
                self.clients_info[cpt][2],  # bandwidth distribution
                self.clients_info[cpt][3],  # number of cores
                self.clients_info[cpt][4],  # number of cores
                self.clients_info[cpt][5]/10,  # frequency distribution
                self.clients_info[cpt][6],  # bandwidth distribution
                self.clients_info[cpt][7],
                self.clients_info[cpt][8],

            )
            self.list_clients.append(client)
            cpt += 1


    def weight_scalling_factor(self, client, active_clients):
        """
        Determine the weight scaling factor for a given client.

        Parameters:
        - client: The client for which to calculate the weight scaling factor.
        - active_clients: List of active clients participating in the current round.

        Returns:
        - The weight scaling factor for the given client.
        """
        # First, calculate the total training data points across clients for this round
        global_count = sum([client_obj.get_size() for client_obj in active_clients])
        
        # Get the total number of data points held by a client
        local_count = client.get_size()
        
        return local_count / global_count
    
    def scale_model_weights(self, weight, scalar):
        """
        Scale the local model parameters of each client by a given scalar factor.

        Parameters:
        - weight: The local model parameters.
        - scalar: The scaling factor.

        Returns:
        - The scaled local model parameters.
        """
        w_scaled = copy.deepcopy(weight)
        
        # Scale each parameter by the given scalar factor
        for k in weight.keys():
            w_scaled[k] = scalar * w_scaled[k]
            
        return w_scaled

    
    def sum_scaled_weights(self, scaled_weight_list):
        """
        Aggregate the different scaled model parameters.

        Parameters:
        - scaled_weight_list: A list of scaled models parameters.

        Returns:
        - The aggregated global model parameters.
        """
        w_avg = copy.deepcopy(scaled_weight_list[0])
        
        for k in w_avg.keys():
            tmp = torch.zeros_like(scaled_weight_list[0][k], dtype=torch.float32)
            
            # Sum up the scaled weights for each client
            for i in range(len(scaled_weight_list)):
                tmp += scaled_weight_list[i][k]
            
            w_avg[k].copy_(tmp)
            
        return w_avg
 

    def flatten(self, weight):
        """
        Flatten the model parameters.

        Parameters:
        - weight: The model parameters.

        Returns:
        - A flattened list of model parameters.
        """
        weight_flatten = []

        # Iterate through the parameters and flatten them
        for param in weight.values():
            weight_flatten.append(np.array(param).reshape(-1))

        # Flatten the list of flattened parameters
        weight_flatten = [item for sublist in weight_flatten for item in sublist]

        return weight_flatten

    
    def select_active_clients_random(self, comm_round, C):
        """
        Select a fraction of clients randomly for a training round.

        Parameters:
        - comm_round: The current communication round.
        - C: The fraction of clients to be selected.

        Returns:
        - A list of randomly selected client indices.
        """
        client_index = np.arange(0, len(self.list_clients))
        
        # Calculate the number of active clients for this round
        m = int(max(C * self.N, 1))  # max between C * N and 1
        
        # Randomly select a fraction of clients
        active_clients = random.sample(list(client_index), k=m)
        
        # Shuffle the list of active clients
        random.shuffle(active_clients)
        
        return active_clients
    
    def calculate_energy_consumption(self,active_clients_index):
        """
        计算各个客户端设备的能耗

        Returns:
        - 一个包含每个客户端能耗的列表，长度等于客户端数量
        """
        energy_consumption = []
        

        for index in range(0,len(self.list_clients)):
            # 获取客户端的训练数据量、频率、带宽等信息
            total_energy=0
            if index in active_clients_index:
                total_energy =  self.list_clients[index].CalculationTrainEngry()+self.list_clients[index].CalculationTranslationEngry(self.bandwidth/10)
            
            # 将计算结果存入列表
            energy_consumption.append(total_energy)
        return energy_consumption  # 列表长度与客户端数量相同

    def global_train(self, comms_round, C, E, mu, M, omega, batch_size, verbose_test=1, verbos=0):
        """
        Perform global training for a federated learning scenario with reinforcement learning aspects.

        Parameters:
        - comms_round: The total number of communication rounds.
        - C: The fraction of clients participating in each round.
        - E: Number of local training iterations on each client.
        - mu: A parameter used for FedProx.
        - M: A parameter used for FAVOR.
        - omega: A parameter used for FAVOR.
        - batch_size: The size of the batch.
        - verbose_test: Verbosity level for printing test information (default is 1).
        - verbos: Verbosity level for printing training information (default is 0).
        - type_data: Type of data, e.g., "Fall" or "others" (default is "others").

        Returns:
        - A dictionary containing various results including the best model weights, accuracy, loss, time, reputation, rewards, and DQL loss.
        """ 
        
        m = int(max(C*self.N, 1))
        total_energy_consumption = np.zeros(len(self.list_clients))

        rounds = []
        accuarcy = []
        loss = []
        reputation_list = []
        list_loss_DQL = []

        time_rounds = []
        time_rounds_sum = []
        best_model_weights = {}
        best_accuracy = 0
        rewards = []
                
        # Initialize the first state
        weight_list_for_iteration = []
        # Append weight of global model
        weight_list_for_iteration.append(self.flatten(self.model.state_dict()))
        
        max_latency = 0
        min_latency = 100000000
        
        
        # For each client perform one epoch of SGD to get the weights
        for client in self.list_clients:
            frequency_client = client.frequency
            bandwidth_client = client.bandwidth
    
            client_w_for_first_iteration = client.train(self.model.state_dict(), 1, mu, verbos)
            weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
            
        # Apply PCA
        pca = PCA(n_components = len(self.list_clients))

    
        # get the weight with PCA
        weight_list_for_iteration_pca = pca.fit_transform(weight_list_for_iteration)

        # State est une concaténation des different weight
        state = torch.Tensor([item for sublist in weight_list_for_iteration_pca for item in sublist])
        
        # init dql
        dql = DQL.DQL(len(state), len(self.list_clients), batch_size)
        
        self.model.train()

        
        # For each communication round 
        for comm_round in  tqdm(range(comms_round)):
            temps_debut = timeit.default_timer()
            rounds.append(comm_round+1)
            
            if (verbos == 1):
                print("*************************************************************************************")
                print("Communication round n : ", comm_round + 1)
                
            # Receive the parameters of the global model (first we will have the initialized parameters)
            global_weights = self.model.state_dict()
            
            # Verify if we need to update the target network
            if (comm_round + 1) % dql.update_rate == 0:
                dql.update_target_network()
            
            
            if (comm_round == 0):
                # random selection
                active_clients_index = self.select_active_clients_random(comm_round, C)
            else:
                # multinomial_selection
                active_clients_index = dql.multiaction_selection(state, C, comm_round, mode = "Mode2")
                
            print(active_clients_index)
                
            # List to collect the parameters of the model * weight of each client
            scaled_local_weight_list = []
            
            # list of active client objetcs
            active_clients = [self.list_clients[i] for i in active_clients_index]

            time_roundt = []
            # For each selected customer
            for client_index in  active_clients_index:
                if (verbos == 1):
                        print("Entrainnement locale du client : ", client.client_name)
                
                # Training on local client data
                client_w = self.list_clients[client_index].train(global_weights, E, mu , verbos)
                
                # Update the reduced weights
                weight_list_for_iteration_pca[client_index] =  (pca.transform(np.array(self.flatten(copy.deepcopy(client_w))).reshape(1, -1)))[0]
                
                # Get the parameters of the local model * factor
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients) #nk/n'
                
                # save in the list
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor) 
                scaled_local_weight_list.append(client_scaled_weight) 
                
                frequency_client = self.list_clients[client_index].frequency
                bandwidth_client = self.list_clients[client_index].bandwidth
                latency_client =  self.list_clients[client_index].CalculationTrainTime()+self.list_clients[client_index].CalculationTranslationTime(self.bandwidth/10)    
                round_energy_consumption = self.calculate_energy_consumption(active_clients_index)  
                total_energy_consumption += np.array(round_energy_consumption)
                #print("Client name : ", self.list_clients[client_index].client_name, " with latency :", latency_client)
                time_roundt.append(latency_client)

            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            
            # The average of the different points received
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)

            # Copy the weights in the global model
            self.model.load_state_dict(average_weights)

            acc_test, loss_test = self.test()
        
            # Test the global model each round
            if (verbose_test == 1):
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(acc_test.numpy()*100, 2) ,", Test loss :", round(loss_test, 2))
                print("*************************************************************************************")
                
            if (acc_test > best_accuracy):
                best_accuracy = acc_test
                best_model_weights = copy.deepcopy(average_weights)
                    
            accuarcy.append(acc_test.item())
            loss.append(loss_test)
            
            
            # Update reduced global parameters
            weight_list_for_iteration_pca[0] =  (pca.transform(np.array(self.flatten(copy.deepcopy(self.model.state_dict()))).reshape(1, -1)))[0]
            
            # Next state
            next_state = torch.Tensor([item for sublist in weight_list_for_iteration_pca for item in sublist])
            
            # We train using one action
            action = active_clients_index[0]
   
            # calcul du reward
            reward = M**(acc_test - omega) -1
            
            rewards.append(reward)
            
            #store the transition information   
            dql.store_transistion(state, action, reward, next_state, done = False)
            #update current state to next state
            state = copy.deepcopy(next_state)

            loss_dql = dql.train(comm_round, mode = "Mode2")
            
            list_loss_DQL.append(loss_dql)
            temps_fin = timeit.default_timer() - temps_debut

         # Cleanup   
        for client in self.list_clients:
            client.delete_model()  

        # Results   
        dict_result = {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuarcy,
            "Loss" : loss,
            "Timeurounds" : time_rounds,
            "Reputation" : reputation_list,
            "Rewards" : rewards,
            "EnergyConsumption" : total_energy_consumption.tolist()  
        }

        return dict_result
    
    def global_train_fall(self, comms_round, C, E, mu, M, omega, batch_size, verbose_test = 1, verbos = 0):
        """
        Perform global training for a federated learning scenario with reinforcement learning aspects for fall detection.

        Parameters:
        - comms_round: The total number of communication rounds.
        - C: The fraction of clients participating in each round.
        - E: Number of local training iterations on each client.
        - mu: A parameter used for FedProx.
        - M: A parameter used for FAVOR.
        - omega: A parameter used for FAVOR.
        - batch_size: The size of the batch.
        - verbose_test: Verbosity level for printing test information (default is 1).
        - verbos: Verbosity level for printing training information (default is 0).
        - type_data: Type of data, e.g., "Fall" or "others" (default is "others").

        Returns:
        - A dictionary containing various results including the best model weights, accuracy, loss, time, reputation, rewards, and DQL loss.
        """ 
        
        rounds = []
        accuarcy = []
        recall = []
        precision = []
        loss = []
        f1score = []
        list_loss_DQL = []
        
        
        time_rounds = []
        time_rounds_sum = []
        best_model_weights = {}
        best_f1score = 0
        rewards = []
                
        # Initialize the first state
        weight_list_for_iteration = []
        # Append weight of global model
        weight_list_for_iteration.append(self.flatten(self.model.state_dict()))
        
        # For each client perform one epoch of SGD to get the weights
        for client in self.list_clients: 
            client_w_for_first_iteration = client.train(self.model.state_dict(), 1, mu, verbos)
            weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
            
        # Apply PCA
        pca = PCA(n_components = len(self.list_clients))

    
        # get the weight with PCA
        weight_list_for_iteration_pca = pca.fit_transform(weight_list_for_iteration)

        # State est une concaténation des different weight
        state = torch.Tensor([item for sublist in weight_list_for_iteration_pca for item in sublist])
        
        # init dql
        dql = DQL.DQL(len(state), len(self.list_clients), batch_size)
        
        self.model.train()
        
        # For each communication round 
        for comm_round in  tqdm(range(comms_round)):

            rounds.append(comm_round+1)
            
            if (verbos == 1):
                print("*************************************************************************************")
                print("Communication round n : ", comm_round + 1)
                
            # Receive the parameters of the global model (first we will have the initialized parameters)
            global_weights = self.model.state_dict()
            
            # Verify if we need to update the target network
            if (comm_round + 1) % dql.update_rate == 0:
                dql.update_target_network()
            
            
            if (comm_round == 0):
                # random selection
                active_clients_index = self.select_active_clients_random(comm_round, C)
            else:
                # multinomial_selection
                active_clients_index = dql.multiaction_selection(state, C, comm_round, mode = "Mode2")
                
            # List to collect the parameters of the model * weight of each client
            scaled_local_weight_list = []
            
            # list of active client objetcs
            active_clients = [self.list_clients[i] for i in active_clients_index]

            time_roundt = []
            # For each selected customer
            for client_index in  active_clients_index:
                if (verbos == 1):
                     print("Entrainnement locale du client : ", client.client_name)
                
                # Training on local client data
                client_w = self.list_clients[client_index].train(global_weights, E, mu, verbos)
                
                # Update the reduced weights
                weight_list_for_iteration_pca[client_index] =  (pca.transform(np.array(self.flatten(copy.deepcopy(client_w))).reshape(1, -1)))[0]
                
                # Get the parameters of the local model * factor
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients) #nk/n'
                
                # save in the list
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor) 
                scaled_local_weight_list.append(client_scaled_weight) 

                frequency_client = random.choice(self.list_clients[client_index].frequency)
                bandwidth_client = random.choice(self.list_clients[client_index].bandwidth)
                
                latency_client = (self.list_clients[client_index].get_size()*64*40*20)/(self.list_clients[client_index].numbercores * 1000000 * frequency_client) + (self.number_parameters * 64)/(1000000 * bandwidth_client)
                print("Client name : ", self.list_clients[client_index].client_name, " with latency :", latency_client)
                time_roundt.append(latency_client)

            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            
            # The average of the different points received
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)

            # Copy the weights in the global model
            self.model.load_state_dict(average_weights)

            acc_test, f1score_test, recall_test, precision_test, loss_test = self.test()
        
            # Test the global model each round
            if (verbose_test == 1):
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(acc_test*100, 2) ,", Test F1_score :", round(f1score_test*100, 2)," Test loss :", round(loss_test, 2))
                print("Test Recall : ", round(recall_test*100, 2), "Test Precision : ", round(precision_test*100, 2))
                print("*************************************************************************************")
                
            if (f1score_test > best_f1score):
                best_f1score = f1score_test
                best_model_weights = copy.deepcopy(average_weights)
                    
            accuarcy.append(acc_test)
            f1score.append(f1score_test)
            recall.append(recall_test)
            precision.append(precision_test)
            loss.append(loss_test)
         
            
            # Update reduced global parameters
            weight_list_for_iteration_pca[0] =  (pca.transform(np.array(self.flatten(copy.deepcopy(self.model.state_dict()))).reshape(1, -1)))[0]
            
            # Next state
            next_state = torch.Tensor([item for sublist in weight_list_for_iteration_pca for item in sublist])
            
            # We train using one action
            action = active_clients_index[0]

                        
            # calcul du reward
            reward = M**(acc_test - omega) -1
            
            rewards.append(reward)

            #store the transition information   
            dql.store_transistion(state, action, reward, next_state, done = False)
            #update current state to next state
            state = copy.deepcopy(next_state)
    

            loss_dql = dql.train(comm_round, mode = "Mode2")
            list_loss_DQL.append(loss_dql)
                        
        # Cleanup
        for client in self.list_clients:
            client.delete_model()  

        # Results  
        dict_result = {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuarcy,
            "F1_score": f1score,
            "Recall" : recall,
            "Precision" : precision,
            "Loss" : loss,
            "Timeurounds" : time_rounds,
            "Timesum" : time_rounds_sum,
            "Rewards" : rewards,
            "LossDQL" : list_loss_DQL
        }
            
            
        return dict_result

    


    def test(self):
        """
        Evaluate the global model with the test dataset for a task other than fall detection.

        Returns:
        - The accuracy and loss on the test dataset.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Testing
        test_loss = 0
        correct = 0

        # Iterate through the test dataset
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.testdataloader):
                log_probs = self.model(data)
                # Sum up batch loss
                test_loss += torch.nn.functional.cross_entropy(log_probs, target, reduction='sum').item()

                # Get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            # Calculate average test loss and accuracy
            test_loss /= len(self.testdataloader.dataset)
            accuracy = correct / len(self.testdataloader.dataset)

        return accuracy, test_loss

