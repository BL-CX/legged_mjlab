# Copyright (c) 2026-2027 zh
"""Reference terrain geometry and differentiable 0.02 m height queries.

Initialization samples MuJoCo heightfield triangles. Runtime queries clamp to the raster boundary."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np
import torch
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg


def reference_generator(cfg) -> TerrainGeneratorCfg:
    """Reproduce the ri_5036_him generator from legged_wbc_mjlab@4eb8fcf."""
    from mjlab.terrains import BoxRandomGridTerrainCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG

    terrain = deepcopy(ROUGH_TERRAINS_CFG)
    terrain.seed = cfg.terrain_seed
    terrain.size = tuple(cfg.patch_size_m)
    terrain.num_rows, terrain.num_cols = cfg.core_rows, cfg.num_cols
    terrain.border_width = cfg.outer_flat_width_m
    terrain.curriculum = True
    for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv", "wave_terrain", "random_rough"):
        terrain.sub_terrains.pop(name)
    terrain.sub_terrains["box_random_grid"] = BoxRandomGridTerrainCfg(
        proportion=0.05,
        grid_width=0.8,
        grid_height_range=(0.02, 0.10),
        platform_width=2.4,
        border_width=0.25,
        holes=False,
        merge_similar_heights=False,
        height_merge_threshold=0.02,
        max_merge_distance=3,
    )
    stair_dimensions = ((0.18, 1.9), (0.20, 1.75), (0.22, 1.65), (0.25, 1.5))
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        base = terrain.sub_terrains.pop(name)
        for index, (width, border_width) in enumerate(stair_dimensions):
            terrain.sub_terrains[f"{name}_{index}"] = replace(
                base,
                step_height_range=(0.05, 0.18),
                step_width=width,
                border_width=border_width,
                platform_width=2.499999,
                holes=False,
                proportion=0.05,
            )
    terrain.sub_terrains["flat"].proportion = 0.06941176470588235
    return terrain


def clean_look(terrain):
    """Apply the reference ground materials, sky, and fill light."""
    from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg

    terrain.textures = (
        TextureCfg(
            name="sky",
            type="skybox",
            builtin="gradient",
            rgb1=(0.93, 0.95, 0.98),
            rgb2=(0.62, 0.74, 0.87),
            width=512,
            height=3072,
        ),
        TextureCfg(
            name="grid",
            type="2d",
            builtin="checker",
            mark="edge",
            rgb1=(0.66, 0.66, 0.65),
            rgb2=(0.62, 0.62, 0.61),
            markrgb=(0.46, 0.47, 0.50),
            width=512,
            height=512,
        ),
    )
    terrain.materials = (
        MaterialCfg(
            name="ground",
            texture="grid",
            texuniform=True,
            texrepeat=(1.0, 1.0),
            reflectance=0.0,
            geom_names_expr=(".*",),
        ),
    )
    terrain.lights = (
        LightCfg(
            name="fill",
            type="directional",
            pos=(0.0, 0.0, 10.0),
            dir=(0.3, 0.2, -1.0),
            diffuse=(0.12, 0.12, 0.12),
            specular=(0.0, 0.0, 0.0),
            ambient=(0.12, 0.13, 0.15),
            castshadow=False,
        ),
    )
    return terrain


class TerrainMap:
    """Map ten HIM terrain families to columns and six increasing difficulty rows."""

    def __init__(self, cfg):
        self.cfg = deepcopy(cfg)
        self.plane = cfg.terrain == "flat"
        self.size = tuple(cfg.patch_size_m)
        self.generator = None if self.plane else reference_generator(cfg)
        self.family_names = ("flat",) if self.plane else tuple(self.generator.sub_terrains)
        weights = (1.0,) if self.plane else tuple(term.proportion for term in self.generator.sub_terrains.values())
        total_weight = sum(weights)
        self.proportions = tuple(weight / total_weight for weight in weights)
        self.rows, self.cols = (1, 1) if self.plane else (self.generator.num_rows, len(self.family_names))
        self.support_half_extent = np.asarray((self.rows, self.cols)) * self.size / 2 + cfg.outer_flat_width_m
        self.origins = None

    def entity_cfg(self):
        if self.plane:
            return clean_look(TerrainEntityCfg(terrain_type="plane"))
        return clean_look(
            TerrainEntityCfg(
                terrain_type="generator", terrain_generator=deepcopy(self.generator), max_init_terrain_level=None
            )
        )

    def bind(self, origins):
        if self.plane:
            return
        self.origins = origins.detach().clone()

    def spawn_origins(self, family_ids, levels):
        return self.origins[levels, family_ids].clone()


class PlaneSurface:
    """Unbounded plane with zero-valued position derivatives."""

    def __init__(self, device="cpu", height=0.0):
        self.device = torch.device(device)
        self.elevation = float(height)
        self.geometry_hash = hashlib.sha256(f"plane/{self.elevation}".encode()).hexdigest()

    def height(self, xy):
        """Query the plane with tensor operations for compiled imagination."""
        return xy.sum(-1) * 0 + self.elevation


class TerrainGeometry:
    """Bucket static collision surfaces by tile for Torch-only runtime queries."""

    @classmethod
    def from_model(cls, model, device="cpu", *, terrain_map):
        self = cls()
        self.device = torch.device(device)
        self.rows, self.cols = terrain_map.rows, terrain_map.cols
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        terrain_ids = []
        for geom in range(model.ngeom):
            body = model.body(model.geom_bodyid[geom]).name
            if body == "terrain" or body.endswith("/terrain"):
                terrain_ids.append(geom)
        if not terrain_ids:
            raise ValueError("Compiled model has no terrain collision geometry")
        bucket_count = self.rows * self.cols + 1
        boxes = [[] for _ in range(bucket_count)]
        fields = [None] * bucket_count
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "rows": self.rows,
                    "cols": self.cols,
                    "size": terrain_map.size,
                    "support": terrain_map.support_half_extent.tolist(),
                },
                sort_keys=True,
            ).encode()
        )
        half = np.asarray((self.rows, self.cols)) * terrain_map.size / 2
        for geom in terrain_ids:
            kind = model.geom_type[geom]
            position = data.geom_xpos[geom].copy()
            matrix = data.geom_xmat[geom].reshape(3, 3)
            if not np.allclose(matrix, np.eye(3), atol=1e-6):
                raise ValueError("Static terrain query only supports axis-aligned boxes and heightfields")
            dimensions = model.geom_size[geom].copy()
            digest.update(np.asarray([kind], dtype=np.int64).tobytes())
            digest.update(position.tobytes())
            digest.update(dimensions.tobytes())
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                lower, upper = position[:2] - dimensions[:2], position[:2] + dimensions[:2]
                first = np.floor((lower + half + 1e-8) / terrain_map.size).astype(int)
                last = np.floor((upper + half - 1e-8) / terrain_map.size).astype(int)
                value = (*position[:2], *dimensions[:2], position[2] + dimensions[2])
                for row in range(max(0, first[0]), min(self.rows - 1, last[0]) + 1):
                    for col in range(max(0, first[1]), min(self.cols - 1, last[1]) + 1):
                        boxes[row * self.cols + col].append(value)
                if (lower < -half).any() or (upper > half).any():
                    boxes[-1].append(value)
            elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
                field_id = model.geom_dataid[geom]
                nrow, ncol = int(model.hfield_nrow[field_id]), int(model.hfield_ncol[field_id])
                address = model.hfield_adr[field_id]
                heights = model.hfield_data[address : address + nrow * ncol].copy()
                field_size = model.hfield_size[field_id].copy()
                digest.update(field_size.tobytes())
                digest.update(heights.tobytes())
                cell = np.floor((position[:2] + half) / terrain_map.size).astype(int)
                if ((cell < 0) | (cell >= (self.rows, self.cols))).any():
                    raise ValueError("Heightfield centre lies outside the terrain map")
                bucket = cell[0] * self.cols + cell[1]
                if fields[bucket] is not None:
                    raise ValueError("Each terrain tile must contain at most one heightfield")
                fields[bucket] = (position, field_size, nrow, ncol, heights)
            else:
                raise ValueError(f"Unsupported terrain collision geometry: {kind}")
        self.geometry_hash = digest.hexdigest()
        max_boxes = max(max(map(len, boxes)), 1)
        box_data = np.zeros((bucket_count, max_boxes, 5), dtype=np.float64)
        box_data[..., 2:4] = -1
        box_data[..., 4] = -np.inf
        for bucket, values in enumerate(boxes):
            if values:
                box_data[bucket, : len(values)] = values
        self.boxes = torch.from_numpy(box_data).to(device)
        max_samples = max([entry[2] * entry[3] for entry in fields if entry is not None] + [4])
        field_values = np.zeros((bucket_count, max_samples), dtype=np.float64)
        field_info = np.zeros((bucket_count, 7), dtype=np.float64)
        field_info[:, 2:4] = 1
        field_rows = np.full((bucket_count, 2), 2, dtype=np.int64)
        field_present = np.zeros(bucket_count, dtype=bool)
        for bucket, entry in enumerate(fields):
            if entry is None:
                continue
            position, sizes, nrow, ncol, heights = entry
            field_values[bucket, : len(heights)] = heights
            field_info[bucket] = (*position[:2], *sizes[:2], position[2], sizes[2], 1)
            field_rows[bucket] = nrow, ncol
            field_present[bucket] = True
        self.field_values = torch.from_numpy(field_values).to(device)
        self.field_info = torch.from_numpy(field_info).to(device)
        self.field_shapes = torch.from_numpy(field_rows).to(device)
        self.field_present = torch.from_numpy(field_present).to(device)
        return self

    def rasterize(self, xs, ys):
        """Rasterize axis-aligned collision geometry without broadcasting every box to every point."""
        heights = torch.full((ys.numel(), xs.numel()), -torch.inf, dtype=torch.float64)
        # Deduplicate boxes registered in adjacent tiles once during initialization.
        boxes = torch.unique(self.boxes.reshape(-1, 5), dim=0)
        boxes = boxes[boxes[:, 2] >= 0]
        for box in boxes:
            cx, cy, hx, hy, top = box.tolist()
            x0 = int(torch.searchsorted(xs, torch.tensor(cx - hx, dtype=xs.dtype)))
            x1 = int(torch.searchsorted(xs, torch.tensor(cx + hx, dtype=xs.dtype), right=True))
            y0 = int(torch.searchsorted(ys, torch.tensor(cy - hy, dtype=ys.dtype)))
            y1 = int(torch.searchsorted(ys, torch.tensor(cy + hy, dtype=ys.dtype), right=True))
            region = heights[y0:y1, x0:x1]
            region.copy_(torch.maximum(region, region.new_tensor(top)))
        for bucket in self.field_present.nonzero().flatten().tolist():
            info = self.field_info[bucket]
            x0, y0 = (int(torch.searchsorted(axis, info[dim] - info[dim + 2])) for dim, axis in enumerate((xs, ys)))
            x1, y1 = (
                int(torch.searchsorted(axis, info[dim] + info[dim + 2], right=True))
                for dim, axis in enumerate((xs, ys))
            )
            gx, gy = torch.meshgrid(xs[x0:x1], ys[y0:y1], indexing="xy")
            points = torch.stack((gx.flatten(), gy.flatten()), -1)
            # Use triangular interpolation to match MuJoCo heightfield collision surfaces.
            values = torch.cat([self._field_heights(chunk, bucket) for chunk in points.split(65536)])
            region = heights[y0:y1, x0:x1]
            region.copy_(torch.maximum(region, values.reshape(region.shape)))
        return torch.where(torch.isfinite(heights), heights, 0).float()

    def _field_heights(self, points, bucket):
        info, sizes = self.field_info[bucket], self.field_shapes[bucket]
        uv = ((points - info[:2] + info[2:4]) / (2 * info[2:4])).clamp(0, 1)
        coordinate = uv * (sizes.flip(-1) - 1)
        index = torch.minimum(coordinate.floor().long(), sizes.flip(-1) - 2)
        fx, fy = (coordinate - index).unbind(-1)
        c, r = index.unbind(-1)
        width = sizes[1]
        z00, z01 = self.field_values[bucket, r * width + c], self.field_values[bucket, r * width + c + 1]
        z10, z11 = self.field_values[bucket, (r + 1) * width + c], self.field_values[bucket, (r + 1) * width + c + 1]
        value = torch.where(
            fx >= fy, z00 + fx * (z01 - z00) + fy * (z11 - z01), z00 + fx * (z11 - z10) + fy * (z10 - z00)
        )
        return info[4] + info[5] * value


class TerrainSurface:
    """Reference 0.02 m raster with bilinear interpolation and boundary clamping."""

    @classmethod
    def from_model(cls, model, device="cpu", *, terrain_map):
        if terrain_map.plane:
            return PlaneSurface(device)
        self = cls()
        self.device = torch.device(device)
        cfg = terrain_map.cfg
        origins = terrain_map.origins.reshape(-1, 3).cpu().numpy()
        self.origin = (float(origins[:, 0].min() - cfg.field_half_m), float(origins[:, 1].min() - cfg.field_half_m))
        upper = (float(origins[:, 0].max() + cfg.field_half_m), float(origins[:, 1].max() + cfg.field_half_m))
        self.res = cfg.field_resolution_m
        geometry = TerrainGeometry.from_model(model, terrain_map=terrain_map)
        payload = f"reference_heightfield_v1/{geometry.geometry_hash}/{self.origin}/{upper}/{self.res}"
        self.geometry_hash = hashlib.sha256(payload.encode()).hexdigest()
        from pathlib import Path

        path = Path(cfg.field_cache_dir) / f"heightfield_blind_{self.geometry_hash[:12]}.pt"
        if path.exists():
            self.h = torch.load(path, map_location=self.device, weights_only=True)["h"]
            return self
        xs = torch.from_numpy(np.arange(self.origin[0], upper[0] + 1e-9, self.res))
        ys = torch.from_numpy(np.arange(self.origin[1], upper[1] + 1e-9, self.res))
        heights = geometry.rasterize(xs, ys)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"h": heights, "origin": self.origin, "res": self.res}, path)
        self.h = heights.to(self.device)
        return self

    def height(self, xy):
        gx = (xy[..., 0] - self.origin[0]) / self.res
        gy = (xy[..., 1] - self.origin[1]) / self.res
        height, width = self.h.shape
        gx, gy = gx.clamp(0, width - 1.001), gy.clamp(0, height - 1.001)
        x0, y0 = gx.floor().long(), gy.floor().long()
        fx, fy = gx - x0, gy - y0
        h00, h01 = self.h[y0, x0], self.h[y0, x0 + 1]
        h10, h11 = self.h[y0 + 1, x0], self.h[y0 + 1, x0 + 1]
        return (h00 * (1 - fx) + h01 * fx) * (1 - fy) + (h10 * (1 - fx) + h11 * fx) * fy
