from abc import ABCMeta, abstractmethod


class EpsDec(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def eps_dec_type(self, epsilon, eps_min, n_games, eps_dec):
        pass


class EpsDecTypeOne(EpsDec):
    def __init__(self) -> None:
        pass

    def eps_dec_type(self, epsilon, eps_min, n_games, eps_dec):
        if epsilon > eps_min:
            epsilon -= 2 / (n_games)
        else:
            epsilon = eps_min

        return epsilon


class EpsDecTypeTwo(EpsDec):
    def __init__(self) -> None:
        pass

    def eps_dec_type(self, epsilon, eps_min, n_games, eps_dec):
        epsilon = epsilon - eps_dec \
            if epsilon > eps_min else eps_min

        return epsilon
