# DRL4SAO

## Abstract

Engineering software systems in an uncertain and ever-changing operating environment is a challenging task. Uncertainty is a cross-cutting phenomenon and system objectives may be jeopardized if it is not handled properly. Uncertainty in self-adaptive systems is any deviation of deterministic knowledge that may reduce the confidence of adaptation decisions made based on the knowledge. Self-adaptation is one prominent method to deal with uncertainty. When system objectives are violated, the self-adaptive system has to analyze all the available adaptation options, i.e., the adaptation space, and choose the best one. Yet analyzing the whole adaptation space using rigorous methods is time-consuming and computationally expensive. Recently machine learning based methods have been proposed to reduce the adaptation space. However, most of them require domain expertise to perform feature engineering and also labeled training data that are representative of the system environment, which may be challenging to obtain due to design-time uncertainty. To tackle these limitations, we present “Deep Reinforcement learning for selecting appropriate adaptation option” - DRL4SAO in short. DRL4SAO uses a deep reinforcement learning based method that can support multiple type of adaptation goals: threshold, set-point and optimization goals. We evaluate the proposed method on two instances of an Internet-of-things application with varying adaptation space sizes. We compare DRL4SAO with (1) a state-of-the-art method that uses deep learning to reduce adaptation options, (2) a baseline that applies exhaustive analysis and (3) one random method that randomly selects an adaptation option. Results show that DRL4SAO is effective with a negligible effect on the realization of the adaptation goals compared to an exhaustive analysis method.

# Installation

Follow These steps to experiment with DRL4SAO

# Clone the repository
```bash
git clone https://github.com/alirezakavianifar/RL-DeltaIoT.git
```
## Create a Virtual Environment in the working directory

```bash
python -m venv venv
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Run the following command

```bash
python main.py
```

