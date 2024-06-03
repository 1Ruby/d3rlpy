import argparse
import pandas as pd
import d3rlpy
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
import numpy as np
import gym
from gym.wrappers import RecordVideo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="halfcheetah-medium-replay-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--use_state_coverage", type=bool, default=True)
    gamma = 0.99
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])
    rl_encoder = d3rlpy.models.encoders.VectorEncoderFactory([400, 300])

    config = d3rlpy.algos.BCQConfig(
        actor_encoder_factory=rl_encoder,
        actor_learning_rate=1e-3,
        critic_encoder_factory=rl_encoder,
        critic_learning_rate=1e-3,
        imitator_encoder_factory=vae_encoder,
        imitator_learning_rate=1e-3,
        batch_size=100,
        lam=0.75,
        action_flexibility=0.05,
        n_action_samples=100,
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
            bcq = config.create(device=args.gpu)
            bcq.fit(
                replay_buffer,
                n_steps=500000,
                n_steps_per_epoch=1000,
                save_interval=100,
                evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
                experiment_name=f"BCQ_{args.dataset}_{args.seed}_state_{i}",
            )
            d3rlpy.notebook_utils.start_virtual_display()
            eval_env = RecordVideo(gym.make("HalfCheetah-v2", render_mode="rgb_array"), f'./video/bcq_halfcheetah_trim_state_{i}')
            reward = d3rlpy.metrics.evaluate_qlearning_with_environment(bcq, eval_env)
            print(reward)
            results.append(reward)
        result_df = pd.DataFrame(columns=['trim_perc', 'reward'])
        result_df['trim_perc'] = np.arange(0, 100, 10)
        result_df['reward'] = results
        result_df.to_csv(f'bcq_{args.dataset}_{args.seed}_state_results.csv')
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
            bcq = config.create(device=args.gpu)
            bcq.fit(
                replay_buffer,
                n_steps=500000,
                n_steps_per_epoch=1000,
                save_interval=100,
                evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
                experiment_name=f"BCQ_{args.dataset}_{args.seed}_reward_{i}",
            )
            d3rlpy.notebook_utils.start_virtual_display()
            env = RecordVideo(gym.make("Halfcheetah-v2", render_mode="rgb_array"), f'./video/bcq_halfcheetah_trim_reward_{i}')
            reward = d3rlpy.metrics.evaluate_qlearning_with_environment(bcq, env)
            print(reward)
            results.append(reward)
        result_df = pd.DataFrame(columns=['trim_perc', 'reward'])
        result_df['trim_perc'] = np.arange(0, 100, 10)
        result_df['reward'] = results
        result_df.to_csv(f'bcq_{args.dataset}_{args.seed}_reward_results.csv')

if __name__ == "__main__":
    main()
