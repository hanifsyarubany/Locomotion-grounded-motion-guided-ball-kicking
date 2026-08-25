from holosoma.config_types.simulator import (
    MujocoBackend,
    PhysxConfig,
    SceneConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
)

isaacgym = SimulatorConfig(
    _target_="holosoma.simulator.isaacgym.isaacgym.IsaacGym",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="isaacgym",
        sim=SimEngineConfig(
            fps=200,
            control_decimation=4,
            substeps=1,
            physx=PhysxConfig(
                solver_type=1,
                num_position_iterations=8,
                num_velocity_iterations=4,
                bounce_threshold_velocity=0.5,
            ),
        ),
        contact_sensor_history_length=3,
    ),
)


isaacsim = SimulatorConfig(
    _target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="isaacsim",
        scene=SceneConfig(
            replicate_physics=True,
        ),
        sim=SimEngineConfig(
            fps=200,
            control_decimation=4,
            substeps=1,
            physx=PhysxConfig(
                solver_type=1,
                # Raised 8->16 / 4->8 (2026-07-17): an open-loop PD-target replay (identical
                # commands, zero policy, IsaacSim vs MuJoCo -- see memory
                # stagec-kick-open-loop-physics-proof) showed the two engines' foot<->floor contact
                # resolution diverging within the very first control tick of the Stage-C kick's
                # single-support phase. More solver iterations means PhysX's own TGS solve is more
                # tightly converged each substep -- reduces PhysX's own numerical contact slop,
                # narrowing (not guaranteed to eliminate) the room for it to diverge from MuJoCo's
                # comparatively well-converged solve. Pure accuracy-for-compute tradeoff, safe on
                # its own terms regardless of whether it closes the gap.
                num_position_iterations=16,
                num_velocity_iterations=8,
                bounce_threshold_velocity=0.5,
            ),
            render_mode="human",
            render_interval=4,
        ),
        contact_sensor_history_length=3,
    ),
)


mujoco = SimulatorConfig(
    _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="mujoco",
        scene=SceneConfig(
            replicate_physics=True,
        ),
        sim=SimEngineConfig(
            fps=200,
            control_decimation=4,
            substeps=1,
            physx=PhysxConfig(
                solver_type=1,
                num_position_iterations=4,
                num_velocity_iterations=0,
                bounce_threshold_velocity=0.5,
            ),
            render_mode="fake",
            render_interval=1,
        ),
        mujoco_backend=MujocoBackend.CLASSIC,  # Explicit for clarity
    ),
)


mjwarp = SimulatorConfig(
    _target_="holosoma.simulator.mujoco.mujoco.MuJoCo",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="mujoco",
        scene=SceneConfig(
            replicate_physics=True,
        ),
        sim=SimEngineConfig(
            fps=200,
            control_decimation=4,
            substeps=1,
            physx=PhysxConfig(
                solver_type=1,
                num_position_iterations=4,
                num_velocity_iterations=0,
                bounce_threshold_velocity=0.5,
            ),
            render_mode="fake",
            render_interval=1,
        ),
        mujoco_backend=MujocoBackend.WARP,  # GPU-accelerated backend
    ),
)


DEFAULTS = {
    "isaacgym": isaacgym,
    "isaacsim": isaacsim,
    "mujoco": mujoco,
    "mjwarp": mjwarp,
}
