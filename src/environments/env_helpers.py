from abc import ABCMeta, abstractmethod
import numpy as np

class IRewardMCOne(metaclass=ABCMeta):
    @abstractmethod
    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None):
        pass

    
class IRewardForHindsight(metaclass=ABCMeta):
    @abstractmethod
    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None):
        pass

class RewardMcOne(IRewardMCOne):
    def __init__(self) -> None:
        self.ut = 9999

    def get_reward(self, util=None, desired_util=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
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
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        if ut <= self.ut:
            reward = 1.0
            self.ut = ut
        else:
            reward = -0.02

        return reward


class RewardMcThree(IRewardMCOne):
    def __init__(self) -> None:
        pass



    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        # Calculate individual penalty/reward for each criterion
        packet_loss_penalty = max(0, packet_loss - packet_thresh)
        latency_penalty = max(0, latency - latency_thresh)

        # Calculate a penalty for being outside the energy consumption range
        energy_penalty = 0
        if float(energy_consumption) < (energy_thresh - 0.1):
            energy_penalty = (energy_thresh - 0.1) - energy_consumption
        elif float(energy_consumption) > (energy_thresh + 0.1):
            energy_penalty = energy_consumption - (energy_thresh + 0.1)

        # Calculate overall reward by combining penalties
        overall_reward = 1.0 - (packet_loss_penalty + latency_penalty + energy_penalty)
        overall_reward = 1 / (1 + np.exp(-overall_reward))


        return overall_reward


class RewardMcFour(IRewardMCOne):
    def __init__(self) -> None:
        pass

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        if ((packet_loss < packet_thresh) and
                (latency < latency_thresh)):
            reward = 1.0
        else:
            reward = -0.02

        return reward


class RewardMcFive(IRewardMCOne):
    def __init__(self) -> None:
        self.energy_consumption = 9999

    def get_reward(self, ut=None, energy_consumption=None, packet_loss=None, latency=None,
                   energy_thresh=None, packet_thresh=None, latency_thresh=None):
        if ((packet_loss < packet_thresh) and
                (latency < latency_thresh) and
                (energy_consumption < self.energy_consumption)):
            self.energy_consumption = energy_consumption
            reward = 1.0
        else:
            reward = -0.02

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
