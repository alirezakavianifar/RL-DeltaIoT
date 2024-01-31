from abc import ABCMeta, abstractmethod
import numpy as np

class IRewardMCOne(metaclass=ABCMeta):
    @abstractmethod
    def get_reward(self, util=None, desired_util=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None):
        pass

    
class IRewardForHindsight(metaclass=ABCMeta):
    @abstractmethod
    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None):
        pass

class RewardMcOne(IRewardMCOne):
    def __init__(self) -> None:
        self.ut = 9999

    def get_reward(self, util=None, desired_util=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None):
        self.ut = desired_util
        if util <= desired_util:
            reward = 1.0
        else:
            reward = -0.02

        return reward


class RewardMcTwo(IRewardMCOne):
    def __init__(self) -> None:
        self.ut = 9999

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None):
        # Define constants
        GOAL_REWARD = 100.0
        TIME_STEP_PENALTY = 0

        # Calculate distance between current position and goal position
        distance_to_goal = abs(energy_consumption - energy_thresh)

        # Positive reward for reaching the goal, inversely proportional to distance
        if energy_consumption == energy_thresh:
            return GOAL_REWARD
        elif distance_to_goal > 0:
            goal_reward = GOAL_REWARD / distance_to_goal
        else:
            goal_reward = GOAL_REWARD


        # Small negative reward for each time step to encourage efficiency
        return goal_reward + TIME_STEP_PENALTY


class RewardMcThree(IRewardMCOne):
    def __init__(self) -> None:
        pass



    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None):
        # Calculate individual penalty/reward for each criterion
        packet_loss_penalty = max(0, packet_loss - packet_thresh)
        latency_penalty = max(0, latency - latency_thresh)

        # Calculate a penalty for being outside the energy consumption range
        energy_penalty = 0
        if float(energy_consumption) < (energy_thresh - setpoint_thresh):
            energy_penalty = (energy_thresh - setpoint_thresh) - energy_consumption
        elif float(energy_consumption) > (energy_thresh + setpoint_thresh):
            energy_penalty = energy_consumption - (energy_thresh + setpoint_thresh)

        # Calculate overall reward by combining penalties
        overall_reward = 1.0 - (packet_loss_penalty + latency_penalty + energy_penalty)
        overall_reward = 1 / (1 + np.exp(-overall_reward))


        return overall_reward


class RewardMcFour(IRewardMCOne):
    def __init__(self) -> None:
        pass

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None):
        # Define weights for each objective
        weight_packet_loss = 1.0
        weight_latency = 1.0

        # Calculate the deviation from the target values
        deviation_packet_loss = max(0, packet_loss - packet_thresh)  # Ensure it's non-negative
        deviation_latency = max(0, latency - latency_thresh)  # Ensure it's non-negative

        # Calculate the overall reward using a weighted sum
        reward = (
            weight_packet_loss * (1 - deviation_packet_loss / packet_thresh) +
            weight_latency * (1 - deviation_latency / latency_thresh))

        return reward


class RewardMcFive(IRewardMCOne):
    def __init__(self) -> None:
        self.energy_consumption = 9999

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None, setpoint_thresh=None):
        # Define weights for each objective
        weight_packet_loss = 1.0
        weight_latency = 1.0
        weight_energy = -1.0  # Negative weight to minimize energy consumption

        # Calculate the deviation from the target values
        deviation_packet_loss = max(0, packet_loss - packet_thresh)  # Ensure it's non-negative
        deviation_latency = max(0, latency - latency_thresh)  # Ensure it's non-negative
        deviation_energy = max(0, energy_consumption - setpoint_thresh)  # Ensure it's non-negative

        # Calculate the overall reward using a weighted sum
        reward = (
            weight_packet_loss * (1 - deviation_packet_loss / packet_thresh) +
            weight_latency * (1 - deviation_latency / latency_thresh) +
            weight_energy * (1 - deviation_energy / (energy_thresh + setpoint_thresh))
        )

        return reward


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
