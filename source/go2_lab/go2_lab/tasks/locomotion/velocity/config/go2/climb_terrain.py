import isaaclab.terrains as terrain_gen
import inspect
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from isaaclab.terrains.trimesh import mesh_terrains_cfg as mtc
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshGapTerrainCfg,
    MeshRailsTerrainCfg,
    MeshPitTerrainCfg,
    MeshBoxTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshStarTerrainCfg,
    MeshRepeatedObjectsTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
)

# 1) sub_terrains 직접 정의
sub_terrains = {
    # 1. Plane
    "plane": MeshPlaneTerrainCfg(
        proportion=0.0,
    ),

    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=(0.05, 0.35),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
    ),

    # 4. Random Grid
    "randomgrid": MeshRandomGridTerrainCfg(
        proportion=0.1,
        grid_width=0.45,
        grid_height_range=(0.1, 0.3),
        platform_width=2.0,
        holes=True,
    ),

    "gap": MeshGapTerrainCfg(
        proportion=0.05,
        gap_width_range=(0.1, 0.5),
        platform_width=1.0,
    ),

    # 5. Rails
    "rails": MeshRailsTerrainCfg(
        proportion=0.2,
        rail_thickness_range=(0.6, 0.6),
        rail_height_range=(0.1, 0.4),
        platform_width=2.0,
    ),

    # 6. Pit
    "pit": MeshPitTerrainCfg(
        proportion=0.2,
        pit_depth_range=(0.1, 0.4),
        platform_width=2.0,
        double_pit=True,
    ),

    # 7. Box
    "box": MeshBoxTerrainCfg(
        proportion=0.2,
        box_height_range=(0.1, 0.4),
        platform_width=2.0,
        double_box=True,
    ),

    # 9. Floating Ring
    "floatingring": MeshFloatingRingTerrainCfg(
        proportion=0.0,
        ring_width_range=(0.6, 0.6),
        ring_height_range=(0.02, 0.2),
        ring_thickness=0.02,
        platform_width=2.0,
    ),
}

# 2) 이 sub_terrains 딕셔너리를 사용해 TerrainGeneratorCfg/ImporterCfg 정의
CLIMB_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,                # curriculum 단계 수
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.01,
    use_cache=False,
    sub_terrains=sub_terrains,
    curriculum=True,
)