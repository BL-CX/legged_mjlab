"""HIMLoco-style terrain preset using mjlab 1.2.0 generators.

The original Isaac Gym setup uses smooth slopes, rough slopes, up/down stairs,
and discrete obstacles with proportions ``[0.1, 0.2, 0.3, 0.3, 0.1]``.
mjlab does not provide HIMLoco's combined slope-plus-noise generator, so the
rough-slope allocation uses its native random height-field generator.
"""

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    # Smooth slope: positive and inverted columns together make 10%.
    "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.05,
      slope_range=(0.0, 0.4),
      platform_width=3.0,
      border_width=0.25,
    ),
    "smooth_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.05,
      slope_range=(0.0, 0.4),
      platform_width=3.0,
      border_width=0.25,
      inverted=True,
    ),
    # Closest native equivalent to HIMLoco's slope-plus-noise terrain.
    "rough_slope": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.2,
      noise_range=(0.01, 0.08),
      noise_step=0.005,
      border_width=0.25,
    ),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.3,
      step_height_range=(0.05, 0.23),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.3,
      step_height_range=(0.05, 0.23),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
      proportion=0.1,
      obstacle_width_range=(1.0, 2.0),
      obstacle_height_range=(0.05, 0.15),
      num_obstacles=20,
      platform_width=3.0,
      border_width=0.25,
    ),
  },
  add_lights=True,
)
