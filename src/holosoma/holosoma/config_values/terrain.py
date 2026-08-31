from holosoma.config_types.multi_skill import load_multi_skill_config, multi_skill_mode_enabled
from holosoma.config_types.simulator import load_ball_config
from holosoma.config_types.terrain import MeshType, TerrainManagerCfg, TerrainTermCfg

# Kick-mode terrain tier + eligibility (2026-08-27), read from the task config the same way every
# other task-config-driven preset in config_values/ does (see unified/g1/randomization.py and
# unified/g1/command.py for the identical module-level pattern): task config in N-skill mode,
# legacy BallConfig otherwise. Resolved once at import time.
#
# Only terrain_unified_mix below consumes these -- the locomotion-only and plane presets are left
# untouched, since kick mode does not exist in those experiments.
_multi_skill_cfg = load_multi_skill_config() if multi_skill_mode_enabled() else None
if _multi_skill_cfg is not None:
    _light_rough_proportion = _multi_skill_cfg.kick_terrain_light_rough_proportion
    _light_rough_max_height = _multi_skill_cfg.kick_terrain_light_rough_max_height
    _kick_eligible_terrain_types = _multi_skill_cfg.kick_eligible_terrain_types
else:
    _legacy_ball_cfg = load_ball_config()
    _light_rough_proportion = _legacy_ball_cfg.kick_terrain_light_rough_proportion
    _light_rough_max_height = _legacy_ball_cfg.kick_terrain_light_rough_max_height
    _kick_eligible_terrain_types = _legacy_ball_cfg.kick_eligible_terrain_types

# terrain_unified_mix takes light_rough's share out of flat's (see its terrain_config below), so
# the proportion cannot exceed flat's own baseline. Guarded here rather than in the yaml loaders
# because this 0.4 is the constant that defines the ceiling, and it lives here. Without this a
# larger value would drive flat negative, which _initialize_terrain_config's `if v > 0.0` filter
# would silently DROP -- yielding a terrain bank with no flat tiles at all and no error.
_UNIFIED_MIX_FLAT_BASELINE = 0.4
if _light_rough_proportion > _UNIFIED_MIX_FLAT_BASELINE:
    raise ValueError(
        f"kick_terrain_light_rough_proportion ({_light_rough_proportion}) exceeds terrain_unified_mix's "
        f"flat share ({_UNIFIED_MIX_FLAT_BASELINE}), which it is carved out of. Lower it, or "
        "rebalance terrain_unified_mix's terrain_config explicitly."
    )

terrain_locomotion_plane = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.PLANE,
        horizontal_scale=1.0,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        max_slope=0.3,
        platform_size=2.0,
        step_width_range=[0.30, 0.40],
        amplitude_range=[0.01, 0.05],
        slope_treshold=0.75,
    )
)

terrain_locomotion_mix = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        terrain_config={
            "flat": 0.2,
            "rough": 0.6,
            "low_obstacles": 0.2,
            "smooth_slope": 0.0,
            "rough_slope": 0.0,
        },
        max_slope=0.3,
        slope_treshold=0.75,
    )
)

# Same generator as terrain_locomotion_mix, but with the "flat" proportion raised — flat terrain
# is the only terrain kick-mode episodes can ever be assigned to (see
# UnifiedManager/env_terrain_is_flat), so the stock 20% flat would cap the achievable kick rate
# at 20% no matter how high configs/skill_mix.yaml's kick_probability goes. Still keeps a majority
# of envs on rough/obstacle terrain for locomotion diversity.
terrain_unified_mix = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=40,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        terrain_config={
            # light_rough's share is taken OUT OF flat's, not appended -- proportions are
            # normalized (Terrain._initialize_terrain_config), so appending would silently dilute
            # rough/low_obstacles and change what LOCOMOTION-mode envs train on. Taking it from
            # flat keeps locomotion's rough/obstacle exposure fixed and only converts some
            # already-flat tiles into gently-uneven ones. At the default proportion of 0.0 this is
            # exactly the original {flat: 0.4, ...} dict, and the zero-proportion entry is filtered
            # out before generation -- a byte-identical no-op.
            "flat": 0.4 - _light_rough_proportion,
            "light_rough": _light_rough_proportion,
            "rough": 0.45,
            "low_obstacles": 0.15,
            "smooth_slope": 0.0,
            "rough_slope": 0.0,
        },
        light_rough_max_height=_light_rough_max_height,
        kick_eligible_terrain_types=_kick_eligible_terrain_types,
        max_slope=0.3,
        slope_treshold=0.75,
    )
)

terrain_load_obj = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        mesh_type=MeshType.LOAD_OBJ,
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        obj_file_path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_parkour.obj",
    )
)

DEFAULTS = {
    "terrain_locomotion_plane": terrain_locomotion_plane,
    "terrain_locomotion_mix": terrain_locomotion_mix,
    "terrain_unified_mix": terrain_unified_mix,
    "terrain_load_obj": terrain_load_obj,
}
