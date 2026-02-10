"""Pi 12-DOF robot constants.

This module defines constants and configuration for the Pi robot,
a 12-DOF bipedal humanoid used in the HoST standing-up task.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# Path resolution.
##

# __file__ is at: HoST/src/host_mjlab/robots/pi_12dof/pi_constants.py
# parents[4] goes up to HoST/
_HOST_ROOT = Path(__file__).resolve().parents[4]

##
# Joint configuration.
##

PI_JOINT_NAMES = (
  "l_hip_pitch_joint",
  "l_hip_roll_joint",
  "l_thigh_joint",
  "l_calf_joint",
  "l_ankle_pitch_joint",
  "l_ankle_roll_joint",
  "r_hip_pitch_joint",
  "r_hip_roll_joint",
  "r_thigh_joint",
  "r_calf_joint",
  "r_ankle_pitch_joint",
  "r_ankle_roll_joint",
)

PI_LEFT_LEG_JOINTS = (
  "l_hip_pitch_joint",
  "l_hip_roll_joint",
  "l_thigh_joint",
  "l_calf_joint",
  "l_ankle_pitch_joint",
  "l_ankle_roll_joint",
)

PI_RIGHT_LEG_JOINTS = (
  "r_hip_pitch_joint",
  "r_hip_roll_joint",
  "r_thigh_joint",
  "r_calf_joint",
  "r_ankle_pitch_joint",
  "r_ankle_roll_joint",
)

# Default joint positions (radians).
PI_DEFAULT_JOINT_POS = {
  "l_hip_pitch_joint": -0.05,
  "l_hip_roll_joint": 0.0,
  "l_thigh_joint": 0.0,
  "l_calf_joint": 0.3,
  "l_ankle_pitch_joint": -0.1,
  "l_ankle_roll_joint": 0.0,
  "r_hip_pitch_joint": -0.05,
  "r_hip_roll_joint": 0.0,
  "r_thigh_joint": 0.0,
  "r_calf_joint": 0.3,
  "r_ankle_pitch_joint": -0.1,
  "r_ankle_roll_joint": 0.0,
}

##
# PD gains from pi_config_ground.py.
##

PI_STIFFNESS = {
  ".*hip_pitch.*": 30.0,
  ".*hip_roll.*": 15.0,
  ".*thigh.*": 15.0,
  ".*calf.*": 30.0,
  ".*ankle_pitch.*": 18.0,
  ".*ankle_roll.*": 8.0,
}

PI_DAMPING = {
  ".*hip_pitch.*": 1.5,
  ".*hip_roll.*": 1.5,
  ".*thigh.*": 1.5,
  ".*calf.*": 1.5,
  ".*ankle_pitch.*": 1.5,
  ".*ankle_roll.*": 1.5,
}

##
# Initial state quaternions for 4-direction reset.
# Format: (w, x, y, z) - MuJoCo quaternion convention.
##

PI_SUPINE_QUAT = (0.7071068, 0.0, -0.7071068, 0.0)
PI_PRONE_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)
PI_LEFT_SIDE_QUAT = (0.7071068, 0.7071068, 0.0, 0.0)
PI_RIGHT_SIDE_QUAT = (0.7071068, -0.7071068, 0.0, 0.0)

PI_INITIAL_ORIENTATIONS = {
  "supine": PI_SUPINE_QUAT,
  "prone": PI_PRONE_QUAT,
  "left_side": PI_LEFT_SIDE_QUAT,
  "right_side": PI_RIGHT_SIDE_QUAT,
}

##
# Action scale.
##

PI_ACTION_SCALE = 1.0

##
# URDF path.
##

PI_URDF_PATH = (
  _HOST_ROOT
  / "legged_gym"
  / "resources"
  / "robots"
  / "pi_12dof"
  / "urdf"
  / "pi_12dof_release_v1.urdf"
)


def _convert_urdf_to_mjcf() -> mujoco.MjSpec:
  """Convert Pi URDF to MuJoCo spec using mujoco.compile()."""
  if not PI_URDF_PATH.exists():
    raise FileNotFoundError(f"Pi URDF not found at {PI_URDF_PATH}")

  # Load URDF and convert to MjSpec.
  # MuJoCo handles URDF conversion automatically, URDF uses radians by default.
  spec = mujoco.MjSpec.from_file(str(PI_URDF_PATH))

  # Add a freejoint to make this a floating-base robot.
  # The root body is bodies[1] (bodies[0] is the worldbody).
  root_body = spec.bodies[1] if len(spec.bodies) > 1 else None
  if root_body is not None:
    # Check if a freejoint already exists.
    has_freejoint = any(jnt.type == mujoco.mjtJoint.mjJNT_FREE for jnt in spec.joints)
    if not has_freejoint:
      root_body.add_freejoint(name="root_freejoint")

  return spec


def get_spec() -> mujoco.MjSpec:
  """Get MuJoCo spec for Pi robot."""
  return _convert_urdf_to_mjcf()


##
# Actuator configuration using IdealPdActuator.
##

PI_ACTUATOR_HIP_PITCH = IdealPdActuatorCfg(
  target_names_expr=(".*hip_pitch_joint",),
  stiffness=30.0,
  damping=1.5,
  effort_limit=20.0,
)

PI_ACTUATOR_HIP_ROLL = IdealPdActuatorCfg(
  target_names_expr=(".*hip_roll_joint",),
  stiffness=15.0,
  damping=1.5,
  effort_limit=20.0,
)

PI_ACTUATOR_THIGH = IdealPdActuatorCfg(
  target_names_expr=(".*thigh_joint",),
  stiffness=15.0,
  damping=1.5,
  effort_limit=20.0,
)

PI_ACTUATOR_CALF = IdealPdActuatorCfg(
  target_names_expr=(".*calf_joint",),
  stiffness=30.0,
  damping=1.5,
  effort_limit=20.0,
)

PI_ACTUATOR_ANKLE_PITCH = IdealPdActuatorCfg(
  target_names_expr=(".*ankle_pitch_joint",),
  stiffness=18.0,
  damping=1.5,
  effort_limit=20.0,
)

PI_ACTUATOR_ANKLE_ROLL = IdealPdActuatorCfg(
  target_names_expr=(".*ankle_roll_joint",),
  stiffness=8.0,
  damping=1.5,
  effort_limit=20.0,
)

##
# Collision configuration.
##

PI_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim=3,
  friction=(0.8, 0.005, 0.0001),
  solref=(0.002, 1.0),
  solimp=(0.98, 0.999, 0.001, 0.5, 2.0),
)

# Foot-specific collision with torsional + rolling friction (condim=6).
# In PhysX, rigid contacts make tiptoe naturally unstable (small support polygon).
# In MuJoCo, soft contacts with condim=3 allow stable tiptoe balance.
# Adding rolling friction (condim=6) penalizes small contact patches:
#   - Flat foot (9 distributed spheres): large rolling resistance torque -> stable
#   - Tiptoe (3 clustered front spheres): small rolling resistance torque -> unstable
PI_FOOT_COLLISION = CollisionCfg(
  geom_names_expr=(".*ankle_roll.*",),
  condim=6,
  friction=(0.8, 0.1, 0.01),  # tangential, torsional (0.1m arm), rolling (0.01m arm)
  solref=(0.002, 1.0),
  solimp=(0.98, 0.999, 0.001, 0.5, 2.0),
  disable_other_geoms=False,  # Don't disable non-foot geoms.
)

##
# Keyframe configuration.
##

PI_LYING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.351),
  rot=PI_SUPINE_QUAT,  # Default to supine.
  joint_pos=PI_DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Articulation configuration.
##

PI_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PI_ACTUATOR_HIP_PITCH,
    PI_ACTUATOR_HIP_ROLL,
    PI_ACTUATOR_THIGH,
    PI_ACTUATOR_CALF,
    PI_ACTUATOR_ANKLE_PITCH,
    PI_ACTUATOR_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_pi_robot_cfg() -> EntityCfg:
  """Get a fresh Pi robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=PI_LYING_KEYFRAME,
    collisions=(PI_COLLISION, PI_FOOT_COLLISION),
    spec_fn=get_spec,
    articulation=PI_ARTICULATION,
  )
