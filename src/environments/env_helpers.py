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
                
            elif goal in ["energy", "packet", "latency", "multi"]:
                return np.array([energy_consumption, packet_loss, latency])
                    
            elif goal == "packet_loss":
                if packet_loss is not None:
                    return packet_loss
                
            elif goal == "latency":
                return latency
            
            elif goal in ["energy_thresh", "packet_thresh", "latency_thresh"]:
                ...
                # energy_consumption = 1.0 if energy_consumption <= energy_thresh else -0.2
                # packet_loss = 1.0 if packet_loss <= packet_thresh else -0.2
                # latency = 1.0 if latency <= latency_thresh else -0.2

                return np.array([energy_consumption, packet_loss, latency])
                    
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

