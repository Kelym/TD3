import numpy as np
import torch
import gym
import argparse
import os
import datetime
import pickle
import yaml

from functools import partial

import utils
import TD3
import OurDDPG
import DDPG

from torch.utils.tensorboard import SummaryWriter

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, create_env_fn, seed, eval_episodes=10):
  eval_env = create_env_fn()
  eval_env.seed(seed + 100)

  avg_reward = 0.
  last_rew = 0.
  for _ in range(eval_episodes):
    state, done = eval_env.reset(), False
    while not done:
      action = policy.select_action(np.array(state))
      state, reward, done, _ = eval_env.step(action)
      avg_reward += reward
    last_rew += reward

  avg_reward /= eval_episodes
  last_rew /= eval_episodes

  #print("---------------------------------------")
  print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, last_rew {last_rew:.2f}")
  #print("---------------------------------------")
  return avg_reward, last_rew


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
  parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
  parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
  parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
  parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
  parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
  parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
  parser.add_argument("--buffer_size", default=100000)            # Buffer Size
  parser.add_argument("--discount", default=0.99)                 # Discount factor
  parser.add_argument("--tau", default=0.005)                     # Target network update rate
  parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
  parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
  parser.add_argument("--actor_lr", default=0.0003)               # Actor learning rate
  parser.add_argument("--critic_lr", default=0.0003)              # Critic learning rate
  parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
  parser.add_argument("--train_step", default=1, type=int)        # Train the agent X times per policy step
  parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
  parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
  parser.add_argument("--affix", default="")

  args = parser.parse_args()
  current_time = datetime.datetime.now()
  file_name = f"{args.env}_{current_time.strftime('%m%d%H%M')}_{args.affix}"
  print("---------------------------------------")
  print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
  print("---------------------------------------")

  if "tycho" in args.env.lower():
    from tycho_env import TychoEnv
    #create_env_fn = partial(TychoEnv,
    #  config={
    #    "state_space": "eepose-obj",
    #    "action_space": "xyz-vel"})
    create_env_fn = partial(TychoEnv,
      config={
        "state_space": "eepose-obj",
        "action_space": "xyz"})
  else:
    create_env_fn = partial(gym.make, args.env)

  env = create_env_fn()

  # Set seeds
  env.seed(args.seed)
  env.action_space.seed(args.seed)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]
  min_action = env.action_space.low
  max_action = env.action_space.high
  half_range_action = (max_action - min_action) / 2

  kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "min_action": min_action,
    "max_action": max_action,
    "discount": args.discount,
    "tau": args.tau,
  }

  # Initialize policy
  if args.policy == "TD3":
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = float(args.policy_noise) * half_range_action
    kwargs["noise_clip"] = float(args.noise_clip) * half_range_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["critic_lr"] = float(args.critic_lr)
    kwargs["actor_lr"] = float(args.actor_lr)
    policy = TD3.TD3(**kwargs)
  elif args.policy == "OurDDPG":
    policy = OurDDPG.DDPG(**kwargs)
  elif args.policy == "DDPG":
    policy = DDPG.DDPG(**kwargs)

  if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    policy.load(f"./models/{policy_file}")

  replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(args.buffer_size),
                                     reserve_size=args.start_timesteps) # SANCHECK allow reserve

  # Evaluate untrained policy
  print("Evaluating untrained")
  evaluations = [eval_policy(policy, create_env_fn, args.seed)]
  print("....")

  summaryWriter = SummaryWriter(f"runs/{file_name}")

  with open(f'runs/{file_name}/args.yml', 'w') as outfile:
    yaml.dump(args, outfile, default_flow_style=False)


  state, done = env.reset(), False
  episode_reward = 0
  episode_timesteps = 0
  episode_num = 0

  for t in range(int(args.max_timesteps)):

    episode_timesteps += 1
    # Select action randomly or according to policy
    if t < args.start_timesteps:
      action = env.action_space.sample()
      #action = np.array(state[-3:]) # SANCHECK: bootstrap with demo
    else:
      action = (
        policy.select_action(np.array(state))
        #np.array(state[-3:]) # SANCHECK: freeze policy to be optimal
        + np.random.normal(0, half_range_action * float(args.expl_noise), size=action_dim)
      ).clip(min_action, max_action)

    # Perform action
    next_state, reward, done, _ = env.step(action)
    done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

    # Store data in replay buffer
    #if t < args.start_timesteps: # SANCHECK freeze buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)
    #if t < args.start_timesteps: # SANCHECK preserve demo
    #  replay_buffer.add_reserve(state, action, next_state, reward, done_bool)
    #else:
    #  replay_buffer.push_with_reserve(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    if done:
      # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
      #print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} LastRew: {reward:.2f}")

      summaryWriter.add_scalar("Rollout/Steps", episode_timesteps, episode_num)
      summaryWriter.add_scalar("Rollout/Rew", episode_reward, episode_num)
      summaryWriter.add_scalar("Rollout/LastRew", reward, episode_num)

      # Reset environment
      state, done = env.reset(), False
      episode_reward = 0
      episode_timesteps = 0
      episode_num += 1

    # Train agent after collecting sufficient data

    #if t == args.start_timesteps:
    #  with open(f"./models/{file_name}-start", "wb") as picklef:
    #    pickle.dump(replay_buffer, picklef)

    if t >= args.start_timesteps:
      for i in range(args.train_step):
        policy.train(replay_buffer, args.batch_size, summaryWriter)
    #else: # SANCHECK: bootstrap with demo
    #  policy.train_actor(replay_buffer, args.batch_size, summaryWriter)
    #  policy.train_critic(replay_buffer, args.batch_size, summaryWriter)

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
      eval_result = eval_policy(policy, create_env_fn, args.seed)
      evaluations.append(eval_result)
      np.save(f"./results/{file_name}", evaluations)
      if args.save_model: policy.save(f"./models/{file_name}")
      summaryWriter.add_scalar("Test/AvgRew", eval_result[0], t+1)
      summaryWriter.add_scalar("Test/LastRew", eval_result[1], t+1)

#with open(f"./models/{file_name}-end", "wb") as picklef:
#  pickle.dump(replay_buffer, picklef)
