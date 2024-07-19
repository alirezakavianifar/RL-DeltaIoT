from abc import ABCMeta, abstractmethod
import numpy as np

class IRewardStrategy(metaclass=ABCMeta):
    @abstractmethod
    def get_reward(self, util=None, desired_util=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None, goal=None):
        pass

class RewardStrategy(IRewardStrategy):
    def __init__(self, strategy_type):
        self.strategy_type = strategy_type
        self.ut = 9999
        self.previous_performance = 9999

    def get_reward(self, util=None, desired_util=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None, goal=None):
        if self.strategy_type == "min":
            if goal == "utilization":
                self.ut = desired_util
                if util <= desired_util:
                    return 1.0
                else:
                    return -0.02
            elif goal == "energy":
                if energy_consumption is not None:
                    # if energy_consumption <= self.previous_performance:
                    #     self.previous_performance = energy_consumption
                    #     return 1.0
                    # else:
                    #     return -0.02
                    return -energy_consumption
                    
            elif goal == "packet_loss":
                if packet_loss is not None:
                    # if packet_loss <= packet_thresh:
                    #     self.previous_performance = packet_loss
                    #     return 1.0
                    # else:
                    #     return -0.02
                    return -packet_loss
            elif goal == "latency":
                if latency is not None:
                    # if latency <= latency_thresh:
                    #     self.previous_performance = latency
                    #     return 1.0
                    # else:
                    #     return -0.02
                    return -latency
            elif goal == "energy_thresh":
                if energy_consumption is not None and energy_thresh is not None:
                    if energy_consumption <= energy_thresh:
                        return 1.0
                    else:
                        return -0.02
            elif goal == "packet_loss_thresh":
                if packet_loss is not None and packet_thresh is not None:
                    if packet_loss <= packet_thresh:
                        return 1.0
                    else:
                        return -0.02
            elif goal == "latency_thresh":
                if latency is not None and latency_thresh is not None:
                    if latency <= latency_thresh:
                        return 1.0
                    else:
                        return -0.02

        elif self.strategy_type == "two":
            if energy_consumption is not None and energy_thresh is not None:
                if energy_consumption >= energy_thresh:
                    return 1.0
                else:
                    return energy_consumption / energy_thresh
            return None

        elif self.strategy_type == "multi":
            if packet_loss is not None and packet_thresh is not None and latency is not None and latency_thresh is not None and energy_consumption is not None and setpoint_thresh is not None:
                packet_loss_penalty = max(0, packet_loss - packet_thresh)
                latency_penalty = max(0, latency - latency_thresh)
                energy_penalty = 0
                if energy_consumption < setpoint_thresh['lower_bound']:
                    energy_penalty = (setpoint_thresh['lower_bound'] - energy_consumption)
                elif setpoint_thresh['lower_bound'] <= energy_consumption <= setpoint_thresh['upper_bound']:
                    energy_penalty = 0
                else:
                    energy_penalty = energy_consumption - setpoint_thresh['upper_bound']
                overall_reward = 1.0 - (packet_loss_penalty + latency_penalty + energy_penalty)
                return 1 / (1 + np.exp(-overall_reward))
            return None

        elif self.strategy_type == "four":
            if packet_loss is not None and packet_thresh is not None and latency is not None and latency_thresh is not None:
                weight_packet_loss = 0.5
                weight_latency = 0.5
                deviation_packet_loss = max(0, packet_loss - packet_thresh)
                deviation_latency = max(0, latency - latency_thresh)
                return (
                    weight_packet_loss * (1 - deviation_packet_loss / packet_thresh) +
                    weight_latency * (1 - deviation_latency / latency_thresh))
            return None

        elif self.strategy_type == "five":
            if packet_loss is not None and packet_thresh is not None and latency is not None and latency_thresh is not None and energy_consumption is not None and energy_thresh is not None and setpoint_thresh is not None:
                weight_packet_loss = 1.0
                weight_latency = 1.0
                weight_energy = -1.0
                deviation_packet_loss = max(0, packet_loss - packet_thresh)
                deviation_latency = max(0, latency - latency_thresh)
                deviation_energy = max(0, energy_consumption - setpoint_thresh)
                return (
                    weight_packet_loss * (1 - deviation_packet_loss / packet_thresh) +
                    weight_latency * (1 - deviation_latency / latency_thresh) +
                    weight_energy * (1 - deviation_energy / (energy_thresh + setpoint_thresh))
                )
            return None

class IInterface(metaclass=ABCMeta):
    @abstractmethod
    def hello(self, name):
        pass

class A(IInterface):
    def hello(self, name):
        return f'hello {name} from class A'

class B(IInterface):
    def __init__(self) -> None:
        self.ut = 9999

    def hello(self, name):
        self.ut -= 1
        return f"Hello {name} from class B for {self.ut}"

class Message:
    def __init__(self, data: IInterface) -> None:
        self.data = data

    def send_message(self, message):
        return self.data.hello(message)

if __name__ == '__main__':
    message = Message(B())
    for i in range(12):
        final_message = message.send_message('Alireza')
        print(final_message)

    # Examples of using RewardStrategy
    reward_strategy_one_util = RewardStrategy("one")
    reward_strategy_one_energy = RewardStrategy("one")
    reward_strategy_one_packet_loss = RewardStrategy("one")
    reward_strategy_one_latency = RewardStrategy("one")

    print(reward_strategy_one_util.get_reward(util=50, desired_util=60, goal="utilization"))
    print(reward_strategy_one_energy.get_reward(energy_consumption=30, energy_thresh=50, goal="energy"))
    print(reward_strategy_one_packet_loss.get_reward(packet_loss=2, packet_thresh=5, goal="packet_loss"))
    print(reward_strategy_one_latency.get_reward(latency=10, latency_thresh=15, goal="latency"))
