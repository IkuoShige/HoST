"""PPO with smooth loss for stable training.

Based on HoST's smooth loss implementation which prevents policy from being
too sensitive to observation changes, avoiding noise std explosion.

Adapted for upstream rsl_rl v4+ API (self.actor/self.critic instead of self.policy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class PPOSmooth(PPO):
  """PPO with smooth loss regularization.

  Adds a smooth loss term that penalizes large changes in policy output
  when observations change slightly. This prevents noise std from exploding
  during training.
  """

  def __init__(
    self,
    *args,
    value_smoothness_coef: float = 0.1,
    smoothness_upper_bound: float = 1.0,
    smoothness_lower_bound: float = 0.1,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.value_smoothness_coef = value_smoothness_coef
    self.smoothness_upper_bound = smoothness_upper_bound
    self.smoothness_lower_bound = smoothness_lower_bound
    self._cont_buffer: list[torch.Tensor] = []

  def process_env_step(self, obs, rewards, dones, extras):
    """Store continuation flags for smooth loss, then call parent."""
    super().process_env_step(obs, rewards, dones, extras)
    self._cont_buffer.append((~dones.bool()).float())

  def update(self) -> dict[str, float]:  # noqa: C901
    """PPO update with smooth loss added on top of the standard PPO loss."""
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_smooth_loss = 0
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None

    # Compute smooth loss coefficients (matching Isaac Gym HoST).
    if self.smoothness_lower_bound > 0:
      epsilon = self.smoothness_lower_bound / (
        self.smoothness_upper_bound - self.smoothness_lower_bound
      )
      policy_smooth_coef = self.smoothness_upper_bound * epsilon
      value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef
    else:
      policy_smooth_coef = 0.0
      value_smooth_coef = 0.0

    # Prepare matched (obs, next_obs, cont) pairs for smooth loss.
    # We extract actor and critic obs separately since they may use different groups.
    actor_obs_flat = None
    actor_next_obs_flat = None
    critic_obs_flat = None
    critic_next_obs_flat = None
    cont_flat = None
    if (
      self._cont_buffer
      and policy_smooth_coef > 0
      and hasattr(self.storage, "observations")
    ):
      num_transitions = len(self._cont_buffer)
      if num_transitions > 1:
        storage_obs = self.storage.observations
        if hasattr(storage_obs, "keys") and callable(storage_obs.keys):
          # TensorDict: extract obs for each model's obs groups.
          actor_key = self.actor.obs_groups[0]  # e.g. "policy"
          critic_key = self.critic.obs_groups[0]  # e.g. "critic"

          actor_obs = storage_obs[actor_key][:num_transitions]
          if actor_obs.dim() == 4:
            actor_obs = actor_obs.flatten(start_dim=2)
          actor_obs_flat = actor_obs[:-1].flatten(0, 1).detach()
          actor_next_obs_flat = actor_obs[1:].flatten(0, 1).detach()

          critic_obs = storage_obs[critic_key][:num_transitions]
          if critic_obs.dim() == 4:
            critic_obs = critic_obs.flatten(start_dim=2)
          critic_obs_flat = critic_obs[:-1].flatten(0, 1).detach()
          critic_next_obs_flat = critic_obs[1:].flatten(0, 1).detach()
        else:
          # Non-TensorDict: same obs for both.
          obs_all = storage_obs[: num_transitions - 1].flatten(0, 1).detach()
          next_obs_all = storage_obs[1:num_transitions].flatten(0, 1).detach()
          actor_obs_flat = critic_obs_flat = obs_all
          actor_next_obs_flat = critic_next_obs_flat = next_obs_all

        cont_stacked = torch.stack(self._cont_buffer[:-1], dim=0)
        cont_flat = cont_stacked.flatten(0, 1)

    total_samples = actor_obs_flat.shape[0] if actor_obs_flat is not None else 0

    # Mini batch generator.
    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for (
      obs_batch,
      actions_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_actions_log_prob_batch,
      old_mu_batch,
      old_sigma_batch,
      hidden_states_batch,
      masks_batch,
    ) in generator:
      num_aug = 1
      original_batch_size = obs_batch.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      # Symmetric augmentation.
      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        obs_batch, actions_batch = data_augmentation_func(
          obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
        )
        num_aug = int(obs_batch.batch_size[0] / original_batch_size)
        old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
        target_values_batch = target_values_batch.repeat(num_aug, 1)
        advantages_batch = advantages_batch.repeat(num_aug, 1)
        returns_batch = returns_batch.repeat(num_aug, 1)

      # Forward pass through actor and critic.
      self.actor(
        obs_batch,
        masks=masks_batch,
        hidden_state=hidden_states_batch[0],
        stochastic_output=True,
      )
      actions_log_prob_batch = self.actor.get_output_log_prob(actions_batch)
      value_batch = self.critic(
        obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
      )

      mu_batch = self.actor.output_mean[:original_batch_size]
      sigma_batch = self.actor.output_std[:original_batch_size]
      entropy_batch = self.actor.output_entropy[:original_batch_size]

      # Adaptive learning rate via KL divergence.
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            dim=-1,
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      # Surrogate loss.
      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Value function loss.
      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (
          value_batch - target_values_batch
        ).clamp(-self.clip_param, self.clip_param)
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
      )

      # === Smooth loss (HoST-specific) ===
      smooth_loss = torch.tensor(0.0, device=self.device)
      if policy_smooth_coef > 0 and actor_obs_flat is not None and total_samples > 0:
        sample_size = min(original_batch_size, total_samples)
        indices = torch.randperm(total_samples, device=self.device)[:sample_size]
        assert actor_next_obs_flat is not None and cont_flat is not None
        assert critic_obs_flat is not None and critic_next_obs_flat is not None

        cont_smooth = cont_flat[indices]
        mix_weights = cont_smooth * (torch.rand_like(cont_smooth) * 2.0 - 1.0)

        # Actor smooth loss: penalize large action changes for small obs changes.
        a_obs = actor_obs_flat[indices]
        a_next = actor_next_obs_flat[indices]
        a_mix = a_obs + mix_weights.unsqueeze(-1) * (a_next - a_obs)

        actor_normalizer = self.actor.obs_normalizer
        actor_net = self.actor.mlp

        original_actions = actor_net(actor_normalizer(a_obs))
        mixed_actions = actor_net(actor_normalizer(a_mix))

        policy_smooth_loss = torch.square(
          torch.norm(original_actions - mixed_actions, dim=-1)
        ).mean()

        # Critic smooth loss.
        c_obs = critic_obs_flat[indices]
        c_next = critic_next_obs_flat[indices]
        c_mix = c_obs + mix_weights.unsqueeze(-1) * (c_next - c_obs)

        critic_normalizer = self.critic.obs_normalizer
        critic_net = self.critic.mlp

        original_values = critic_net(critic_normalizer(c_obs))
        mixed_values = critic_net(critic_normalizer(c_mix))

        value_smooth_loss = torch.square(
          torch.norm(original_values - mixed_values, dim=-1)
        ).mean()

        smooth_loss = (
          policy_smooth_coef * policy_smooth_loss
          + value_smooth_coef * value_smooth_loss
        )
        loss = loss + smooth_loss

      # Symmetry loss.
      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          obs_batch, _ = data_augmentation_func(
            obs=obs_batch, actions=None, env=self.symmetry["_env"]
          )
          num_aug = int(obs_batch.shape[0] / original_batch_size)

        mean_actions_batch = self.actor(obs_batch.detach().clone())
        action_mean_orig = mean_actions_batch[:original_batch_size]
        _, actions_mean_symm_batch = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
        )

        mse_loss = nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions_batch[original_batch_size:],
          actions_mean_symm_batch.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      # RND loss.
      if self.rnd:
        with torch.no_grad():
          rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
          rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
        predicted_embedding = self.rnd.predictor(rnd_state_batch)
        target_embedding = self.rnd.target(rnd_state_batch).detach()
        mseloss = nn.MSELoss()
        rnd_loss = mseloss(predicted_embedding, target_embedding)

      # Gradient step.
      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd_optimizer:
        self.rnd_optimizer.step()

      # Accumulate losses.
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      mean_smooth_loss += smooth_loss.item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    # Compute means.
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_smooth_loss /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates

    # Free smooth loss data.
    del actor_obs_flat, actor_next_obs_flat
    del critic_obs_flat, critic_next_obs_flat, cont_flat

    self.storage.clear()
    self._cont_buffer.clear()

    loss_dict: dict[str, float] = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "smooth": mean_smooth_loss,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss

    return loss_dict
