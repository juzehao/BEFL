import copy
import timeit
import clientFL.Client as Client
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import torch.nn as nn
import RL.DQL as DQL
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.nn import functional as F

class Server_FLASHRL(object):
    
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
        global_count = sum([client_obj.get_size() for client_obj in active_clients])
        local_count = client.get_size()
        return local_count / global_count

    def scale_model_weights(self, weight, scalar):
        w_scaled = copy.deepcopy(weight)
        for k in weight.keys():
            w_scaled[k] = scalar * w_scaled[k]
        return w_scaled

    def sum_scaled_weights(self, scaled_weight_list):
        w_avg = copy.deepcopy(scaled_weight_list[0])
        for k in w_avg.keys():
            tmp = torch.zeros_like(scaled_weight_list[0][k], dtype=torch.float32)
            for i in range(len(scaled_weight_list)):
                tmp += scaled_weight_list[i][k]
            w_avg[k].copy_(tmp)
        return w_avg
    
    def flatten(self, weight):
        weight_flatten = []
        for param in weight.values():
            weight_flatten.append(np.array(param).reshape(-1))
        weight_flatten = [item for sublist in weight_flatten for item in sublist]
        return weight_flatten

    def flatten_state(self, state_list):
        result_list = []
        max_length = max(len(sublist) for sublist in state_list)
        for i in range(max_length):
            for sublist in state_list:
                if i < len(sublist):
                    element = sublist[i]
                    if isinstance(element, list):
                        result_list.extend(element)
                    else:
                        result_list.append(element)
        return torch.Tensor(result_list)
    def select_active_clients_random(self, comm_round, C):
        '''
        Randomly select a fraction of active clients for a training round.

        Parameters:
        - comm_round: The communication round.
        - C: Fraction of active clients.

        Returns:
        - List of indices representing the selected active clients.
        '''
        client_index = np.arange(0, len(self.list_clients))
        
        m = int(max(C * self.N, 1))  # max between c*k and 1

        active_clients = random.sample(list(client_index), k=m)  # Select a fraction of clients
        
        return active_clients

    def global_train(self, comms_round, C, E, mu, lamb, rep_init, batch_size, verbose_test=1, verbos=0, init_weights=None, init_commround=None, reputation_init=None):
        m = int(max(C * self.N, 1))
        rounds = []
        accuracy = []
        reputation_list = []
        rewards = []
        loss = []
        time_rounds = []
        time_rounds_sum = []
        energy_consumption = np.zeros(len(self.list_clients))
        best_model_weights = {}
        best_accuracy = 0
        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    
        # Initialize the first state
        weight_list_for_iteration = []
        numbersamples_for_iteration = []
        numbercores_for_iteration = []
        frequencies_for_iteration = []
        bandwidth_for_iteration = []
        max_latency = 0
        min_latency = float('inf')
        
        if init_weights is not None:
            reputation_clients_t = reputation_init
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            self.model.load_state_dict(copy.deepcopy(init_weights))
            comm_start = init_commround
            
            for client in self.list_clients:
                client.set_weights(copy.deepcopy(init_weights))
                
            # For each client, perform one epoch of SGD to get the weights
            for client in self.list_clients:
                frequency_client = random.choice(client.frequency)
                bandwidth_client = client.bandwidth
                
                latency_min = client.CpuCycle*client.D/client.numbercores*max(client.frequency) + client.CalculationTranslationTime(self.bandwidth/10)
                latency_max = client.CpuCycle*client.D/client.numbercores*max(client.frequency) + client.CalculationTranslationTime(self.bandwidth/10)

                max_latency = max(latency_max, max_latency)
                min_latency = min(latency_min, min_latency)
                
                client_w_for_first_iteration = client.get_model()
                weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
                numbersamples_for_iteration.append(client.get_size())
                numbercores_for_iteration.append(client.numbercores)
                frequencies_for_iteration.append(frequency_client)
                bandwidth_for_iteration.append(bandwidth_client)
            
        else:
            reputation_clients_t = np.full(self.N, rep_init)
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            comm_start = 0
            
            # For each client, perform one epoch of SGD to get the weights
            for client in self.list_clients:
                frequency_client = random.choice(client.frequency)
                bandwidth_client = client.bandwidth
                latency_min = client.CpuCycle*client.D/client.numbercores*max(client.frequency) + client.CalculationTranslationTime(self.bandwidth/10)
                latency_max = client.CpuCycle*client.D/client.numbercores*max(client.frequency) + client.CalculationTranslationTime(self.bandwidth/10)

                max_latency = max(latency_max, max_latency)
                min_latency = min(latency_min, min_latency)
                
                client_w_for_first_iteration = client.train(self.model.state_dict(), 1, mu, verbos)
                weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
                numbersamples_for_iteration.append(client.get_size())
                numbercores_for_iteration.append(client.numbercores)
                frequencies_for_iteration.append(frequency_client)
                bandwidth_for_iteration.append(bandwidth_client)
        
        # Apply PCA
        pca = PCA(n_components=len(self.list_clients))
        weight_list_for_iteration_pca = pca.fit_transform(weight_list_for_iteration)
        state_list = []

        for cpt in range(len(self.list_clients)):
            client_state = []
            client_state.append(list(weight_list_for_iteration_pca[cpt]))
            client_state.append(numbersamples_for_iteration[cpt])
            client_state.append(numbercores_for_iteration[cpt])
            client_state.append(frequencies_for_iteration[cpt])
            client_state.append(bandwidth_for_iteration[cpt])
            state_list.append(client_state)  
        
        state = self.flatten_state(state_list)
        
        dql = DQL.DQL(len(state), len(self.list_clients), batch_size, flag=bool(init_weights))
        Accuracy_global_previous, loss_test = self.test()
            
        self.model.train()
        list_loss_DQL = []
        
        for comm_round in range(comm_start, comms_round):
            rounds.append(comm_round + 1)
            global_weights = self.model.state_dict()
                
            if (comm_round + 1) % dql.update_rate == 0:
                dql.update_target_network()
                
            if comm_round == 0:
                active_clients_index = self.select_active_clients_random(comm_round, C)
            else:
                active_clients_index = dql.multiaction_selection(state, C, comm_round, mode="Mode1")
                    
            active_clients_index.sort()
            scaled_local_weight_list = []
            active_clients = [self.list_clients[i] for i in active_clients_index]
            weight_local_clients = []
            time_roundt = []

            for client_index in active_clients_index:
                client_w = self.list_clients[client_index].train(global_weights, E, mu, verbos)
                weight_local_clients.append(self.flatten(client_w))
                state_list[client_index][0] = list((pca.transform(np.array(self.flatten(copy.deepcopy(client_w))).reshape(1, -1)))[0])
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients)
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor)
                scaled_local_weight_list.append(client_scaled_weight)

                frequency_client = random.choice(self.list_clients[client_index].frequency)
                bandwidth_client = self.list_clients[client_index].bandwidth
                latency_client = self.list_clients[client_index].CalculationTrainTime()+self.list_clients[client_index].CalculationTranslationTime(self.bandwidth/10)
                state_list[client_index][3] = frequency_client
                state_list[client_index][4] = bandwidth_client
                time_roundt.append(latency_client)
                energy_consumption[client_index] +=   self.list_clients[client_index].CalculationTrainEngry()+self.list_clients[client_index].CalculationTranslationEngry(self.bandwidth/10)
                
            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.model.load_state_dict(average_weights)
            Accuracy_global_t, loss_test = self.test()
            
            if verbose_test == 1:
                print(f"Training round n: {comm_round + 1}, Test accuracy: {round(Accuracy_global_t.numpy() * 100, 2)}, Test loss: {round(loss_test, 2)}")
                print("*************************************************************************************")
                    
            if Accuracy_global_t > best_accuracy:
                best_accuracy = Accuracy_global_t
                best_model_weights = copy.deepcopy(average_weights)
                    
            accuracy.append(Accuracy_global_t)
            loss.append(loss_test)
            
            next_state = self.flatten_state(state_list)
            action = np.zeros(len(self.list_clients))
            action[active_clients_index] = 1
            normalized_distance = 1 / num_param * (np.sum((np.array(weight_local_clients) - np.array(self.flatten(self.model.state_dict()))) / np.array(self.flatten(average_weights)), axis=1))
            utility_clients = np.exp(-np.abs(normalized_distance)) if Accuracy_global_t > Accuracy_global_previous else 1 - np.exp(-np.abs(normalized_distance))
            reputation_clients_t[active_clients_index] = (1 - lamb) * reputation_clients_t[active_clients_index] + lamb * (utility_clients - ((np.array(time_roundt) - min_latency) / (max_latency - min_latency)))
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            reward = np.array(reputation_clients_t[active_clients_index])
            rewards.append(reward)
                
            if comm_round == comms_round - 1:
                dql.store_transistion(state, action, reward, next_state, done=True)
            else:
                dql.store_transistion(state, action, reward, next_state, done=False)
                    
            state = copy.deepcopy(next_state)
            Accuracy_global_previous = Accuracy_global_t
            list_loss_DQL.append(dql.train(comm_round, mode="Mode1"))
                
        for client in self.list_clients:
            client.delete_model()
                
        dict_result = {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuracy,
            "Loss": loss,
            "Timeurounds": time_rounds,
            "Timesum": time_rounds_sum,
            "Reputation": reputation_list,
            "Rewards": rewards,
            "EnergyConsumption": energy_consumption.tolist(),
            "LossDQL": list_loss_DQL
        }

        return dict_result
    
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.testdataloader):
                log_probs = self.model(data)
                test_loss += torch.nn.functional.cross_entropy(log_probs, target, reduction='sum').item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            test_loss /= len(self.testdataloader.dataset)
            accuracy = correct / len(self.testdataloader.dataset)
        return accuracy, test_loss
