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
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score
from torch.nn import functional as F
from scipy.optimize import minimize
global energy
class BEFL_offline(object):
    
    def __init__(self, num_clients, global_model, dict_clients, loss_fct, B, dataset_test, learning_rate, momentum, clients_info,bandwidth,alpha):
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
        self.alpha=alpha
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
            print(client.get_size())
            self.list_clients.append(client)
            cpt += 1
            print()


    def weight_scalling_factor(self, client, active_clients):
        '''
        Determine the factor assigned to a given client.

        Parameters:
        - client: The client for which the factor is determined.
        - active_clients: List of active clients.

        Returns:
        - The scaling factor for the client.
        '''
        # Calculate the total training data points across clients for this round
        global_count = sum([client_obj.get_size() for client_obj in active_clients])
        # Get the total number of data points held by a client
        local_count = client.get_size()

        return local_count / global_count

    def scale_model_weights(self, weight, scalar):
        '''
        Multiply the local parameters of each client by its factor.

        Parameters:
        - weight: The local model weights.
        - scalar: The scaling factor.

        Returns:
        - Scaled local model weights.
        '''
        w_scaled = copy.deepcopy(weight)

        for k in weight.keys():
            w_scaled[k] = scalar * w_scaled[k]

        return w_scaled

    def sum_scaled_weights(self, scaled_weight_list):
        '''
        Aggregate different parameters.

        Parameters:
        - scaled_weight_list: List of scaled models weights.

        Returns:
        - Aggregated weights.
        '''
        w_avg = copy.deepcopy(scaled_weight_list[0])

        for k in w_avg.keys():
            tmp = torch.zeros_like(scaled_weight_list[0][k], dtype=torch.float32)

            for i in range(len(scaled_weight_list)):
                tmp += scaled_weight_list[i][k]

            w_avg[k].copy_(tmp)

        return w_avg
    
    
    def flatten(self, weight):
        '''
        Flatten the parameters of a model.

        Parameters:
        - weight: The model weights.

        Returns:
        - Flattened model weights.
        '''
        weight_flatten = []

        for param in weight.values():
            weight_flatten.append(np.array(param).reshape(-1))

        weight_flatten = [item for sublist in weight_flatten for item in sublist]

        return weight_flatten

    def flatten_state(self, state_list):
        '''
        Flatten a list of states.

        Parameters:
        - state_list: List of states where each state is represented as a list of sublists.

        Returns:
        - Flattened list of states as a torch.Tensor.
        '''
        result_list = []
        max_length = max(len(sublist) for sublist in state_list)  # Find the maximum length of sublists

        for i in range(max_length):
            for sublist in state_list:
                if i < len(sublist):
                    element = sublist[i]
                    if isinstance(element, list):
                        result_list.extend(element)
                    else:
                        result_list.append(element)
        
        return torch.Tensor(result_list)
        
    def calculate_energy_consumption(self,active_clients_index,bandwidth_list):
        """
        计算各个客户端设备的能耗

        Returns:
        - 一个包含每个客户端能耗的列表，长度等于客户端数量
        """
        energy_consumption = []
        

        for index in range(0,len(self.list_clients)):
            # 获取客户端的训练数据量、频率、带宽等信息
            total_energy=0
            # 将计算结果存入列表
            energy_consumption.append(total_energy)
        i=0
        for index in active_clients_index:
            total_energy =  self.list_clients[index].CalculationTrainEngry()+self.list_clients[index].CalculationTranslationEngry(self.bandwidth*bandwidth_list[i])
            i+=1
            energy_consumption[index]=total_energy
        return energy_consumption  # 列表长度与客户端数量相同


    def calculate_value_function(self, client, global_weights):
        """
        Calculate the value function for a client based on the loss of its data.
        
        Parameters:
        - client: The client for which to calculate the value function.
        - global_weights: The current global model weights.
        
        Returns:
        - The calculated value function.
        """
        # Set the client's model to the current global weights
        client.set_weights(global_weights)
        # Calculate the loss on the client's data
        loss = client.calculate_loss()
        # Normalize the loss based on the number of samples
        value = loss / np.sqrt(client.get_size())
        return value


    def choosefromclient(self,client_index,number):
        choose=[]
        for index in client_index:
            choose.append(self.list_clients[index].EnergyFunction(self.bandwidth/10))
        answer_index=np.argsort(choose)[-number:]
        answer=[]
        for index in answer_index:
            answer.append(client_index[index])
        return answer

            

    
    def global_train(self, comms_round, C, E, mu, lamb, rep_init, batch_size, verbose_test=1, verbos=0, init_weights=None, init_commround=None, reputation_init=None):
        m = int(max(C*self.N, 1))
        total_energy_consumption = np.zeros(len(self.list_clients))
        rounds = []
        accuarcy = []
        reputation_list = []
        rewards = []
        loss = []
        time_rounds = [0]
        best_model_weights = {}
        best_accuracy = 0
        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        state_list = []
        high_client=[]
        low_client=[]
        
        if init_weights is not None:
            reputation_clients_t = reputation_init
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            self.model.load_state_dict(copy.deepcopy(init_weights))
            comm_start = init_commround
            for client in self.list_clients:
                client.set_weights(copy.deepcopy(init_weights))
        else:
            reputation_clients_t = np.full(self.N, rep_init)
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            comm_start = 0
            client_energy=[]
            for index in range(len(self.list_clients)):
                client_energy.append(self.list_clients[index].CalculationTranslationEngry(self.bandwidth))
            low_client=np.argsort(client_energy)[:(int(len(self.list_clients)*self.alpha))]
            high_client=np.argsort(client_energy)[(int(len(self.list_clients)*self.alpha)):]
        
        for comm_round in range(comm_start, comms_round):
            rounds.append(comm_round+1)
            global_weights = self.model.state_dict()
            active_clients_index_low=self.choosefromclient(low_client,int(10*self.alpha))
            active_clients_index_high=self.choosefromclient(high_client,10-int(10*self.alpha))
            active_clients_index = active_clients_index_low+active_clients_index_high
            active_clients_index.sort()

            bandwidth_list = []
            num = 10
            for i in range(0, num):
                bandwidth_list.append(1.0 / num)
            roundengry = 0

            energy = []
            for index in active_clients_index:
                energy.append(self.list_clients[index].CalculationTranslationEngry(self.bandwidth))

            def fun(x):
                return sum(e / x_i for e, x_i in zip(energy, x))

            e = 1e-10

            cons1 = (
                {'type': 'eq', 'fun': fun},  # 约束条件：xyz=1
                *[
                    {'type': 'ineq', 'fun': lambda x, i=i: x[i] - e}
                    for i in range(len(energy))
                ],
                *[
                    {'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i]}
                    for i in range(len(energy))
                ],
                {'type': 'ineq', 'fun': lambda x: 1 - sum(x)}  # 新增约束条件：x[i] 的和小于等于 1
            )

            res = minimize(fun, bandwidth_list, method='SLSQP', constraints=cons1)
            bandwidth_list = res.x

            scaled_local_weight_list = []
            active_clients = [self.list_clients[i] for i in active_clients_index]
            t=[]
            index=0
            for client_index in active_clients_index:
                self.list_clients[client_index].count+=1
                client_w = self.list_clients[client_index].train(global_weights, E, mu, verbos)
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients)
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor)
                scaled_local_weight_list.append(client_scaled_weight)
                t.append(self.list_clients[client_index].CalculationTranslationTime(bandwidth_list[index]*self.bandwidth)+self.list_clients[client_index].CalculationTrainTime())
                index+=1
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.model.load_state_dict(average_weights)
            Accuray_global_t, loss_test = self.test()
            if verbose_test == 1:
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(Accuray_global_t.numpy()*100, 2) ,", Test loss :", round(loss_test, 2))
            if Accuray_global_t > best_accuracy:
                best_accuracy = Accuray_global_t
                best_model_weights = copy.deepcopy(average_weights)
            accuarcy.append(Accuray_global_t)
            loss.append(loss_test)
            time_rounds.append(time_rounds[-1]+max(t))
            round_energy_consumption = self.calculate_energy_consumption(active_clients_index,bandwidth_list)  
            total_energy_consumption += np.array(round_energy_consumption)
            comm_round+=1

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