import argparse
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import d3rlpy
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
import gym
from gym.wrappers import RecordVideo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="halfcheetah-medium-replay-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--use_state_coverage", type=bool, default=True)
    args = parser.parse_args()
    gamma = 0.99
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    config = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        weight_temp=3.0,
        max_weight=100.0,
        expectile=0.7,
        reward_scaler=reward_scaler,
    )
    results = []
    acc_rewards = []
    if args.use_state_coverage:
        stds = []
        for episode in dataset.episodes:
            rewards = episode.rewards
            states = np.array(episode.observations)
            acc_reward = 0
            for i in range(len(rewards)):
                acc_reward += rewards[i] * gamma**i
            acc_rewards.append(acc_reward[0])
            std = np.std(states, axis=0)
            stds.append(np.mean(std))

        acc_rewards = np.array(acc_rewards)
        stds = np.array(stds)
        acc_rewards = (acc_rewards - np.mean(acc_rewards)) / np.std(acc_rewards)
        stds = (stds - np.mean(stds)) / np.std(stds)
        score = acc_rewards + stds
        score_sorted = np.sort(score)
        deciles = np.percentile(score_sorted, np.arange(0, 100, 10))
        for i in range(10):
            episodes = [e for j, e in enumerate(dataset.episodes) if score[j] >= deciles[i]]
            buffer = FIFOBuffer(limit=1000000)
            # initialize with pre-collected episodes
            replay_buffer = ReplayBuffer(buffer=buffer, episodes=episodes)
            iql = config.create(device=args.gpu)
            iql.build_with_dataset(replay_buffer)
            assert iql.impl
            scheduler = CosineAnnealingLR(
                iql.impl._modules.actor_optim,  # pylint: disable=protected-access
                500000,
            )

            def callback(algo: d3rlpy.algos.IQL, epoch: int, total_step: int) -> None:
                scheduler.step()

            iql.fit(
                replay_buffer,
                n_steps=500000,
                n_steps_per_epoch=1000,
                save_interval=100,
                callback=callback,
                evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
                experiment_name=f"IQL_{args.dataset}_{args.seed}_state_{i}",
            )
            d3rlpy.notebook_utils.start_virtual_display()
            env = RecordVideo(gym.make("Halfcheetah-v2", render_mode="rgb_array"), f'./video/iql_halfcheetah_trim_state_{i}')
            reward = d3rlpy.metrics.evaluate_qlearning_with_environment(iql, env)
            print(reward)
            results.append(reward)
        result_df = pd.DataFrame(columns=['trim_perc', 'reward'])
        result_df['trim_perc'] = np.arange(0, 100, 10)
        result_df['reward'] = results
        result_df.to_csv(f'iql_{args.dataset}_{args.seed}_state_results.csv')
    else:
        for episode in dataset.episodes:
            rewards = episode.rewards
            acc_reward = 0
            for i in range(len(rewards)):
                acc_reward += rewards[i] * gamma**i
            acc_rewards.append(acc_reward[0])
        acc_rewards_sorted = np.sort(np.array(acc_rewards))
        deciles = np.percentile(acc_rewards_sorted, np.arange(0, 100, 10))
        for i in range(10):
            episodes = [e for j, e in enumerate(dataset.episodes) if acc_rewards[j] >= deciles[i]]
            buffer = FIFOBuffer(limit=1000000)
            # initialize with pre-collected episodes
            replay_buffer = ReplayBuffer(buffer=buffer, episodes=episodes)
            iql = config.create(device=args.gpu)
            iql.build_with_dataset(replay_buffer)
            assert iql.impl
            scheduler = CosineAnnealingLR(
                iql.impl._modules.actor_optim,  # pylint: disable=protected-access
                500000,
            )

            def callback(algo: d3rlpy.algos.IQL, epoch: int, total_step: int) -> None:
                scheduler.step()

            iql.fit(
                replay_buffer,
                n_steps=500000,
                n_steps_per_epoch=1000,
                save_interval=100,
                callback=callback,
                evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
                experiment_name=f"IQL_{args.dataset}_{args.seed}_reward_{i}",
            )
            d3rlpy.notebook_utils.start_virtual_display()
            env = RecordVideo(gym.make("Halfcheetah-v2", render_mode="rgb_array"), f'./video/iql_halfcheetah_trim_reward_{i}')
            reward = d3rlpy.metrics.evaluate_qlearning_with_environment(iql, env)
            print(reward)
            results.append(reward)
        result_df = pd.DataFrame(columns=['trim_perc', 'reward'])
        result_df['trim_perc'] = np.arange(0, 100, 10)
        result_df['reward'] = results
        result_df.to_csv(f'iql_{args.dataset}_{args.seed}_reward_results.csv')

if __name__ == "__main__":
    main()
