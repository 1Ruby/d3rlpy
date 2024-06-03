import argparse
import numpy as np
import d3rlpy
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
import gym
from gym.wrappers import RecordVideo
import pandas as pd

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

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    
    conservative_weight = 5.0

    config = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
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
            cql = config.create(device=args.gpu)

            cql.fit(
                replay_buffer,
                n_steps=500000,
                n_steps_per_epoch=1000,
                save_interval=100,
                evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
                experiment_name=f"CQL_{args.dataset}_{args.seed}_state_{i}",
            )
            cql.save_model(f"CQL_{args.dataset}_{args.seed}_state_{i}.d3")
            d3rlpy.notebook_utils.start_virtual_display()
            eval_env = RecordVideo(gym.make("HalfCheetah-v2", render_mode="rgb_array"), f'./video/cql_halfcheetah_trim_state_{i}')
            reward = d3rlpy.metrics.evaluate_qlearning_with_environment(cql, eval_env)
            print(reward)
            results.append(reward)
        result_df = pd.DataFrame(columns=['trim_perc', 'reward'])
        result_df['trim_perc'] = np.arange(0, 100, 10)
        result_df['reward'] = results
        result_df.to_csv(f'cql_{args.dataset}_{args.seed}_state_results.csv')
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
            cql = config.create(device=args.gpu)
            cql.fit(
                replay_buffer,
                n_steps=500000,
                n_steps_per_epoch=1000,
                save_interval=100,
                evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
                experiment_name=f"CQL_{args.dataset}_{args.seed}_reward_{i}",
            )
            cql.save_model(f"CQL_{args.dataset}_{args.seed}_reward_{i}.d3")
            d3rlpy.notebook_utils.start_virtual_display()
            eval_env = RecordVideo(gym.make("HalfCheetah-v2", render_mode="rgb_array"), f'./video/cql_halfcheetah_trim_state_{i}')
            reward = d3rlpy.metrics.evaluate_qlearning_with_environment(cql, eval_env)
            print(reward)
            results.append(reward)
        result_df = pd.DataFrame(columns=['trim_perc', 'reward'])
        result_df['trim_perc'] = np.arange(0, 100, 10)
        result_df['reward'] = results
        result_df.to_csv(f'cql_{args.dataset}_{args.seed}_reward_results.csv')

if __name__ == "__main__":
    main()
