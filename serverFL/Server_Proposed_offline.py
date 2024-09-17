import copy
import clientFL.Client as Client
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import RL.DQL as DQL
import timeit
import statistics
from scipy.optimize import minimize

class Server_Proposed_offline(object):
    
    def __init__(self, num_clients, global_model, dict_clients, loss_fct, B, dataset_test, learning_rate, momentum, clients_info, bandwidth):
        self.N = num_clients
        self.bandwidth = bandwidth
        self.model = global_model 
        self.list_clients = [] 
        self.B = B
        self.dataset_test = dataset_test
        self.testdataloader = DataLoader(self.dataset_test, batch_size=self.B)
        self.dict_clients = dict_clients
        self.loss_function = copy.deepcopy(loss_fct)
        self.number_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.clients_info = clients_info
        self.create_clients(learning_rate, momentum)

    def create_clients(self, learning_rate, momentum):
        cpt = 0
        for client_name in self.dict_clients.keys():
            client = Client.Client(
                self.dict_clients[client_name],
                copy.deepcopy(self.model),
                copy.deepcopy(self.loss_function),
                learning_rate,
                momentum,
                self.B,
                self.clients_info[cpt][0],
                self.clients_info[cpt][1],
                self.clients_info[cpt][2],
                self.clients_info[cpt][3],
                self.clients_info[cpt][4],
                self.clients_info[cpt][5] / 10,
                self.clients_info[cpt][6],
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

    def calculate_state(self):
        """
        Calculate the state representation for FedRank.
        The state includes data statistics and system characteristics for each client.
        """
        state = []
        for client in self.list_clients:
            # System characteristics
            client.f=random.choice(client.frequency)

            latency = client.CalculationTrainTime() + client.CalculationTranslationTime(self.bandwidth / 10)
            energy_consumption = client.CalculationTrainEngry() + client.CalculationTranslationEngry(self.bandwidth / 10)
            
            # Data statistics
            data_size = client.get_size()
            
            # Combine all these into a single state vector for this client
            client_state = [data_size, latency, energy_consumption,client.total_energy]
            state.extend(client_state)
        
        return torch.Tensor(state)
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
    def calculate_reward(self, previous_accuracy, current_accuracy, total_energy_consumption, total_latency,variance, alpha=0.01, beta=0.01,gamma=0.01):
        """
        Calculate the reward based on accuracy gain, energy consumption, and latency.
        """
        accuracy_gain = current_accuracy - previous_accuracy
        reward = accuracy_gain
        if total_energy_consumption>0.01:
            reward-=(total_energy_consumption-0.01)*alpha
        if total_latency>20:
            reward-=(total_latency-20)*beta
        reward-=gamma*variance
        return reward
    def select_active_clients_offline(self, comm_round, C, alpha=0.5):
        """
        Select a fraction of active clients based on the AFL value function.

        Parameters:
        - comm_round: The communication round.
        - C: Fraction of active clients.
        - alpha1, alpha2, alpha3: Tuning parameters.

        Returns:
        - List of indices representing the selected active clients.
        """
        self.alpha=0.5
        client_energy=[]
        for index in range(len(self.list_clients)):
            client_energy.append(self.list_clients[index].CalculationTranslationEngry(self.bandwidth))
        low_client=np.argsort(client_energy)[:(int(len(self.list_clients)*self.alpha))]
        high_client=np.argsort(client_energy)[(int(len(self.list_clients)*self.alpha)):]
        
        
        global_weights = self.model.state_dict()
        active_clients_index_low=self.choosefromclient(low_client,int(10*self.alpha))
        active_clients_index_high=self.choosefromclient(high_client,10-int(10*self.alpha))
        active_clients_index = active_clients_index_low+active_clients_index_high
        active_clients_index.sort()
        for index in active_clients_index:
            self.list_clients[index].count+=1
        return active_clients_index
    def global_train(self, comms_round, C, E, mu, M, omega, batch_size, imitation_rounds=10, verbose_test=1, verbos=0):
        m = int(max(C * self.N, 1))
        total_energy_consumption = np.zeros(len(self.list_clients))

        rounds = []
        accuracy = []
        loss = []
        list_loss_DQL = []
        time_rounds = []
        time_rounds_sum = []
        best_model_weights = {}
        best_accuracy = 0
        rewards = []
                
        weight_list_for_iteration = []
        weight_list_for_iteration.append(self.flatten(self.model.state_dict()))
        
        for client in self.list_clients:
            client_w_for_first_iteration = client.train(self.model.state_dict(), 1, mu, verbos)
            weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))

        state = self.calculate_state()  # Updated state calculation
        
        dql = DQL.DQL(len(state), len(self.list_clients), batch_size)
        
        self.model.train()

        # **Imitation Learning Phase**
        print("Offline Training using AFL")
        for _ in range(imitation_rounds):
            state = self.calculate_state()  # Update the state before each iteration
            expert_actions = self.select_active_clients_offline(0, C)  # AFL selection instead of random
            for action in expert_actions:
                reward = 1  # High reward for expert actions
                next_state = self.calculate_state()  # State should be updated for each action
                dql.store_transistion(state, action, reward, next_state, done=False)
                dql.train(0, mode="Mode2")  # Train DQL with the imitation data

        # **Online Training Phase** (DQL)
        for comm_round in  tqdm(range(comms_round)):
            temps_debut = timeit.default_timer()
            rounds.append(comm_round+1)
            
            if verbos == 1:
                print("*************************************************************************************")
                print("Communication round n : ", comm_round + 1)
                
            global_weights = self.model.state_dict()
            
            if (comm_round + 1) % dql.update_rate == 0:
                dql.update_target_network()
            
            
            active_clients_index = dql.multiaction_selection(state, C, comm_round, mode="Mode2")
                
            print(active_clients_index)
            scaled_local_weight_list = []
            active_clients = [self.list_clients[i] for i in active_clients_index]

            time_roundt = []
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
            
            print(bandwidth_list)
            scaled_local_weight_list = []
            active_clients = [self.list_clients[i] for i in active_clients_index]
            time_roundt = []
            
            index=0
            nowroundenergy=0

            for client_index in active_clients_index:
                if verbos == 1:
                    print("Training client : ", client_index)
                
                client_w = self.list_clients[client_index].train(global_weights, E, mu, verbos)
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients)
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor) 
                scaled_local_weight_list.append(client_scaled_weight)

                latency_client = self.list_clients[client_index].CalculationTrainTime() + self.list_clients[client_index].CalculationTranslationTime(bandwidth_list[index]*self.bandwidth)
                time_roundt.append(latency_client)
                self.list_clients[client_index].total_energy+=client.CalculationTrainEngry() + client.CalculationTranslationEngry(bandwidth_list[index]*self.bandwidth)
                total_energy_consumption[client_index] += client.CalculationTrainEngry() + client.CalculationTranslationEngry(bandwidth_list[index]*self.bandwidth)
                nowroundenergy+=client.CalculationTrainEngry() + client.CalculationTranslationEngry(bandwidth_list[index]*self.bandwidth)

                index+=1
            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            e=[]
            for client in self.list_clients:
                e.append(client.total_energy)
            variance = statistics.variance(e)

            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.model.load_state_dict(average_weights)

            acc_test, loss_test = self.test()
        
            if verbose_test == 1:
                print("Training round n :", (comm_round+1), ", Test accuracy : ", round(acc_test.numpy()*100, 2), ", Test loss :", round(loss_test, 2))
                print("*************************************************************************************")
                
            if acc_test > best_accuracy:
                best_accuracy = acc_test
                best_model_weights = copy.deepcopy(average_weights)
                    
            accuracy.append(acc_test.item())
            loss.append(loss_test)

            weight_list_for_iteration[0] = self.flatten(copy.deepcopy(self.model.state_dict()))
            next_state = self.calculate_state()  # Updated state calculation
            
            reward = self.calculate_reward(best_accuracy, acc_test, nowroundenergy, max(time_roundt),variance=variance)
            rewards.append(reward)
            
            dql.store_transistion(state, active_clients_index[0], reward, next_state, done=False)
            state = copy.deepcopy(next_state)

            loss_dql = dql.train(comm_round, mode="Mode2")
            list_loss_DQL.append(loss_dql)
            temps_fin = timeit.default_timer() - temps_debut

        for client in self.list_clients:
            client.delete_model()

        return {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuracy,
            "Loss": loss,
            "TimeRounds": time_rounds,
            "Rewards": rewards,
            "EnergyConsumption": total_energy_consumption.tolist()
        }
    
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
