![format check](https://github.com/takuseno/scikit-batch-rl/workflows/format%20check/badge.svg)
![test](https://github.com/takuseno/scikit-batch-rl/workflows/test/badge.svg)

# scikit-batch-rl
Data-driven Deep Reinforcement Learning library in scikit-learn style.
Unlike the other RL libraries, scikit-batch-rl is designed for practical projects.

```py
from skbrl.dataset import MDPDataset
from skbrl.algos import BEAR

# MDPDataset takes arrays of state transitions
dataset = MDPDataset(observations, actions, rewards, terminals)

# train data-driven deep RL
bear = BEAR()
bear.fit(dataset.episodes)

# ready to control
actions = bear.predict(x)
```

## scikit-learn compatibility
This library is designed as if born from scikit-learn.
You can fully utilize scikit-learn's utilities to increase your productivity.
```py
from sklearn.model_selection import train_test_split
from skbrl.metrics.scorer import td_error_scorer

train_episodes, test_episodes = train_test_split(dataset)

bear.fit(train_episodes,
         eval_episodes=test_episodes,
         scorers={'td_error': td_error_scorer})
```

## deploy
Machine learning models often require dependencies even after deployment.
scikit-batch-rl provides more flexible options to solve this problem via torch
script so that the production environment never cares about which algorithms
and hyperparameters are used to train.

```py
# save the learned greedy policy as torch script
bear.save_policy('policy.pt')

# load the policy without any dependencies except pytorch
policy = torch.jit.load('policy.pt')
actions = policy(x)
```

even on C++
```c++
torch::jit::script::Module module;
try {
  module = torch::jit::load('policy.pt');
} catch (const c10::Error& e) {
  //
}
```

## supported algorithms
### discrete control
- [x] DQN
- [x] Double DQN
- [ ] REM

### continuous control
- [x] DDPG
- [x] TD3
- [x] SAC
- [ ] BCQ
- [ ] BEAR
- [ ] MOPO

## supported techniques
- [ ] Quantile Regression
- [ ] Implicit Quantile Network
- [ ] random network augmentation

## supported evaluation metrics
scikit-learn style scoring functions are provided.
```py
from skrbl.metrics.scorer import td_error_scorer

train_episodes, test_episodes = train_test_split(dataset)

bear.fit(train_episodes,
         eval_episodes=test_episodes,
         scorers={'td_error': td_error_scorer})
```

If you have an access to the target environments, you can also evaluate
algorithms on them.
```py
from skbrl.metics.scorer import evaluate_on_environment

env_scorer = evaluate_on_environment(gym.make('Pendulum-v0'))

bear.fit(train_episodes,
         eval_episodes=test_episodes,
         scorers={'td_error': td_error_scorer,
                  'environment': env_scorer})
```

- [ ] Off-policy Classification (requires success flags)
- [x] Temporal-difference Error
- [x] Discounted Sum of Advantages
- [x] Evaluation with gym-like environments

## contributions
### coding style
This library is fully formatted with [yapf](https://github.com/google/yapf).
You can format the entire scripts as follows:
```
$ ./scripts/format
```

### test
The unit tests are provided as much as possible.
You can run the entire tests as follows:
```
$ ./scripts/test
```

If you give `-p` option, the performance tests with toy tasks are also run
(this will take minutes).
```
$ ./scripts/test -p
```
