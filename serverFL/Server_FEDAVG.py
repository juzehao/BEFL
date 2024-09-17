import copy
import clientFL.Client as Client
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.nn import functional as F
import timeit

class Server_FEDAVG(object):

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
        """
        Multiply the local parameters of each client by its factor.

        Parameters:
        - weight: The local model weights.
        - scalar: The scaling factor.

        Returns:
        - Scaled local model weights.
        """
        w_scaled = copy.deepcopy(weight)
        for k in weight.keys():
            w_scaled[k] = scalar * w_scaled[k]
        return w_scaled

    def sum_scaled_weights(self, scaled_weight_list):
        """
        Aggregate different parameters.

        Parameters:
        - scaled_weight_list: List of scaled models weights.

        Returns:
        - Aggregated weights.
        """
        w_avg = copy.deepcopy(scaled_weight_list[0])
        for k in w_avg.keys():
            tmp = torch.zeros_like(scaled_weight_list[0][k], dtype=torch.float32)
            for i in range(len(scaled_weight_list)):
                tmp += scaled_weight_list[i][k]
            w_avg[k].copy_(tmp)
        return w_avg

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
    
    def global_train(self, comms_round, C, E, mu,  rep_init, verbose_test=1, verbos=0, init_weights=None, init_commround=None, reputation_init=None):
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

        for comm_round in range(comm_start, comms_round):
            rounds.append(comm_round+1)
            global_weights = self.model.state_dict()
            active_clients_index = self.select_active_clients_random(comm_round, C)
            active_clients_index.sort()
            scaled_local_weight_list = []
            active_clients = [self.list_clients[i] for i in active_clients_index]
            t=[]
            for client_index in active_clients_index:
                client_w = self.list_clients[client_index].train(global_weights, E, 0.1, verbos)
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients)
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor)
                scaled_local_weight_list.append(client_scaled_weight)
                t.append(self.list_clients[client_index].CalculationTranslationTime(self.bandwidth/10)+self.list_clients[client_index].CalculationTrainTime())
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
            round_energy_consumption = self.calculate_energy_consumption(active_clients_index)  
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
        Evaluate the global model with the test dataset.

        Returns:
        - The accuracy and loss on the test dataset.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.testdataloader:
                log_probs = self.model(data)
                test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            test_loss /= len(self.testdataloader.dataset)
            accuracy = correct / len(self.testdataloader.dataset)
        return accuracy, test_loss
