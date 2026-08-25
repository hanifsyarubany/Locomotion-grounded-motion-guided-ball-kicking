from dataclasses import replace

from holosoma.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig
from holosoma.config_types.video import CartesianCameraConfig

disabled = DisabledLoggerConfig()

wandb = WandbLoggerConfig(mode="online")

wandb_offline = WandbLoggerConfig(mode="offline")

# Stock WBT's default camera (offset=[2,2,1], target_offset=[0,0,0.3]) is framed tight around the
# robot's pelvis. For the ball-kick task the ball sits a few meters away at a configurable
# position (configs/ball.yaml), so that framing routinely leaves it out of the recorded video
# even though it's genuinely present in the sim. Pulled back further and re-aimed slightly ahead
# of the robot (where the ball always is, per ball.yaml's x = forward-from-robot convention) so
# both the robot and the ball area are more likely to stay in frame as the robot moves toward it.
wandb_ball_kick = replace(
    wandb,
    video=replace(
        wandb.video,
        camera=CartesianCameraConfig(offset=[3.5, 3.5, 2.0], target_offset=[1.2, 0.0, 0.3]),
    ),
)

DEFAULTS = {
    "disabled": disabled,
    "wandb": wandb,
    "wandb_offline": wandb_offline,
    "wandb_ball_kick": wandb_ball_kick,
}
