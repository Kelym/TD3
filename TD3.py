import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, min_action, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.min_action = torch.Tensor(min_action).float()
		self.max_action = max_action
		self.half_range_action = torch.Tensor(max_action - min_action).float() / 2
		self.mid_action = torch.Tensor(max_action + min_action).float() / 2

	def to(self, device):
		super().to(device)
		self.min_action = self.min_action.to(device)
		self.half_range_action = self.half_range_action.to(device)
		self.mid_action = self.mid_action.to(device)
		return self

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.half_range_action * torch.tanh(self.l3(a)) + self.mid_action


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		min_action,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		actor_lr=3e-4,
		critic_lr=3e-4,
	):

		self.actor = Actor(state_dim, action_dim, min_action, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.min_action = torch.tensor(min_action, dtype=torch.float32, device=device)
		self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device)
		self.discount = discount
		self.tau = tau
		self.policy_noise = torch.tensor(policy_noise, dtype=torch.float32, device=device)
		self.noise_clip = torch.tensor(noise_clip, dtype=torch.float32, device=device)
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train_actor(self, replay_buffer, batch_size, summaryWriter):

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		if True:
			# Compute actor losse
			#actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			actor_loss = torch.linalg.norm(self.actor(state) - state[:,-3:], dim=1).mean() # SANCHECK: bootstrap with demo
			#target_Q1, target_Q2 = self.critic_target(state, self.actor(state))
			#target_Q = torch.min(target_Q1, target_Q2)
			#actor_loss = -target_Q.mean() # minQforPi

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			#summaryWriter.add_scalar("Train/ActorLoss", actor_loss.item(), self.total_it)


	def train_critic(self, replay_buffer, batch_size, summaryWriter):
		self.total_it += 1
		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
				self.actor_target(next_state) + noise # ? Should we trust the policy decision more as we go?
				#next_state[:,-3:] + noise # SANCHECK bootstrap with optimal policy
			).clamp(self.min_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		summaryWriter.add_scalar("Train/CriticLoss", critic_loss.item(), self.total_it)
		summaryWriter.add_scalar("Train/CriticTarget", target_Q.mean(), self.total_it)

	def train(self, replay_buffer, batch_size, summaryWriter):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
				self.actor_target(next_state) + noise # ? Should we trust the policy decision more as we go?
			).clamp(self.min_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		summaryWriter.add_scalar("Train/CriticLoss", critic_loss.item(), self.total_it)
		summaryWriter.add_scalar("Train/CriticTarget", target_Q.mean(), self.total_it)

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			#actor_loss = torch.linalg.norm(self.actor(state) - state[:,-3:], dim=1).mean() # SANCHECK: optimal critic
			#target_Q1, target_Q2 = self.critic_target(state, self.actor(state))
			#target_Q = torch.min(target_Q1, target_Q2)
			#actor_loss = -target_Q.mean() # minQforPi

			# Optimize the actor SANCHECK freeze actor
			if self.total_it > 0:
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			summaryWriter.add_scalar("Train/ActorLoss", actor_loss.item(), self.total_it)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
