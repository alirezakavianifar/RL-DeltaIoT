# DRL4SAO

# DRL4SAO

Context: Engineering software systems in an uncertain and ever-changing operating environment is a challenging task. Uncertainty is a cross-cutting phenomenon and system objectives may be jeopardized if it is not handled properly. Uncertainty in self-adaptive systems is any deviation of deterministic knowledge that may reduce the confidence of adaptation decisions made based on the knowledge. Self-adaptation is one prominent method to deal with uncertainty. When system objectives are violated, the self-adaptive system has to analyze all the available adaptation options, i.e., the adaptation space, and choose the best one. Yet analyzing the whole adaptation space using rigorous methods is time-consuming and computationally expensive. Recently machine learning based methods have been proposed to reduce the adaptation space. However, most of them require domain expertise to perform feature engineering and also labeled training data that are representative of the system environment, which may be challenging to obtain due to design-time uncertainty.
Objectives: This paper aims to propose a method that can manage uncertainty in self-adaptive systems with large adaptation spaces in an effective and efficient manner.
Method: To tackle this challenge, a method that integrates deep reinforcement learning with self-adaptive systems is developed, considering factors such as scalability, real-time decision-making, and resource constraints.
Result: Results show that the proposed method is effective with a negligible effect on the realization of the adaptation goals compared to a state-of-the-art method.
Conclusion: This article aims to investigate how deep reinforcement learning can be utilized to handle large adaptation spaces in self-adaptive systems efficiently and effectively. By developing a novel method integrating DRL algorithms with self-adaptive systems, this study seeks to improve exploration and exploitation capabilities of self-adaptive systems when dealing with uncertainty.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
