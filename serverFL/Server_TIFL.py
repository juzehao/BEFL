import copy
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.nn import functional as F
import clientFL.Client as Client

class Server_TIFL(object):

    def __init__(self, num_clients, global_model, dict_clients, loss_fct, B, dataset_test, learning_rate, momentum, clients_info, tier_thresholds,bandwidth):
        """
        初始化Server_TIFL对象。

        参数:
        - num_clients: 客户端的数量。
        - global_model: 全局模型。
        - dict_clients: 包含每个客户端信息的字典。
        - loss_fct: 训练使用的损失函数。
        - B: 批处理大小。
        - dataset_test: 用于评估的测试数据集。
        - learning_rate: 训练的学习率。
        - momentum: 训练的动量参数。
        - clients_info: 客户端的信息列表。
        - tier_thresholds: 用于分类客户端的门槛值列表。
        """
        self.N = num_clients
        self.model = global_model
        self.list_clients = []
        self.B = B
        self.bandwidth=bandwidth
        self.dataset_test = dataset_test
        self.testdataloader = DataLoader(self.dataset_test, batch_size=self.B)
        self.dict_clients = dict_clients
        self.loss_function = copy.deepcopy(loss_fct)
        self.number_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.clients_info = clients_info
        self.create_clients(learning_rate, momentum)
        
        # 初始化基于门槛值的层级分类
        self.tier_thresholds = tier_thresholds
        self.client_tiers = self.categorize_clients_into_tiers()

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
    
    def categorize_clients_into_tiers(self):
        """
        根据客户端的性能将其分类到不同的层级中。

        返回:
        - 客户端层级的字典，其中包含客户端的下标。
        """
        client_tiers = {tier: [] for tier in range(len(self.tier_thresholds) + 1)}

        for index, client in enumerate(self.list_clients):
            performance = self.evaluate_client_performance(client)
            tier = self.get_client_tier(performance)
            client_tiers[tier].append(index)  # 将客户端的下标添加到层级中

        return client_tiers

    def evaluate_client_performance(self, client):
        return client.CalculationTrainTime() + client.CalculationTranslationTime(self.bandwidth/10)

    def get_client_tier(self, performance):
        for i, threshold in enumerate(self.tier_thresholds):
            if performance <= threshold:
                return i
        return len(self.tier_thresholds)

    def select_tier(self):
        """
        基于预定义的概率或自适应方法选择一个层级。

        返回:
        - 选择的层级。
        """
        tier_probs = [1 / len(self.client_tiers) for _ in range(len(self.client_tiers))]
        selected_tier = np.random.choice(list(self.client_tiers.keys()), p=tier_probs)
        return selected_tier

    def select_clients_from_tier(self, tier, C):
        """
        从特定层级中选择客户端的下标。

        参数:
        - tier: 要从中选择客户端的层级。
        - C: 选择的客户端比例。

        返回:
        - 选择的客户端下标列表。
        """
        clients_in_tier = self.client_tiers[tier]
        while not clients_in_tier:  # 如果层级为空，则重新选择层级
            tier = self.select_tier()
            clients_in_tier = self.client_tiers[tier]
        
        m = int(max(C * len(clients_in_tier), 1))

        if m > len(clients_in_tier):
            selected_clients = clients_in_tier
        else:
            selected_clients = random.sample(clients_in_tier, m)
        
        return selected_clients



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
            active_clients_index = self.select_clients_from_tier(comm_round, C)
            print(active_clients_index)
            active_clients_index.sort()
            scaled_local_weight_list = []
            active_clients = [self.list_clients[i] for i in active_clients_index]
            t=[]
            for client_index in active_clients_index:
                client_w = self.list_clients[client_index].train(global_weights, E, mu, verbos)
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients)
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor)
                scaled_local_weight_list.append(client_scaled_weight)
                t.append(self.list_clients[client_index].CalculationTranslationTime(self.bandwidth/10)+self.list_clients[client_index].CalculationTrainTime())
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.model.load_state_dict(average_weights)
            Accuray_global_t, loss_test = self.test()
            if verbose_test == 1:
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(Accuray_global_t*100, 2) ,", Test loss :", round(loss_test, 2))
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
        在测试数据集上评估全局模型。

        返回:
        - 测试数据集上的准确率和损失。
        """
        self.model.eval()
        total, correct = 0, 0
        loss_final_test = 0
        y_true = []
        y_pred = []
        for data in self.testdataloader:
            inputs, labels = data
            outputs = self.model(inputs)
            loss_final_test += F.cross_entropy(outputs, labels)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
        accuracy = correct / total
        loss_final_test = loss_final_test / len(self.testdataloader)
        return accuracy, loss_final_test.item()
