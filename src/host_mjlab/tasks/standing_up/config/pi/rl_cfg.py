"""RL training configuration for Pi robot standing-up task."""

from dataclasses import dataclass

from mjlab.rl.config import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class PPOSmoothAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO algorithm config with smooth loss parameters."""

  class_name: str = "host_mjlab.rl.ppo_smooth:PPOSmooth"
  value_smoothness_coef: float = 0.1
  smoothness_upper_bound: float = 1.0
  smoothness_lower_bound: float = 0.1


def make_pi_standing_up_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL training config for Pi robot standing-up task.

  Returns:
    Training configuration with PPO hyperparameters matching HoST.
  """
  return RslRlOnPolicyRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=12000,
    save_interval=100,
    experiment_name="pi_standing_up",
    run_name="",
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=("standing_up", "pi"),
    obs_groups={
      "actor": ("policy",),
      "critic": ("critic",),
    },
    actor=RslRlModelCfg(
      init_noise_std=0.3,
      noise_std_type="log",
      obs_normalization=False,
      hidden_dims=(512, 256, 128),
      activation="elu",
      stochastic=True,
    ),
    critic=RslRlModelCfg(
      obs_normalization=False,
      hidden_dims=(512, 256),
      activation="elu",
      stochastic=False,
    ),
    algorithm=PPOSmoothAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.001,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      value_smoothness_coef=0.1,
      smoothness_upper_bound=1.0,
      smoothness_lower_bound=0.1,
    ),
  )


# Pre-built configuration for convenience.
PI_STANDING_UP_RL_CFG = make_pi_standing_up_rl_cfg()
