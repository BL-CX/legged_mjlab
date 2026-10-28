# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quaternion helpers required by the AMP motion loader."""

import numpy as np
import torch


def QuaternionNormalize(q):
    """Normalize an xyzw quaternion; reject a zero norm."""
    q_norm = np.linalg.norm(q)
    if np.isclose(q_norm, 0.0):
        raise ValueError(
            "Quaternion may not be zero in QuaternionNormalize: |q| = %f, q = %s" % (q_norm, q)
        )
    return q / q_norm


def standardize_quaternion(q):
    """Choose the equivalent quaternion with a nonnegative scalar component."""
    if q[-1] < 0:
        q = -q
    return q


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Interpolate unit quaternions without mutating inputs or producing acos NaNs."""
    eps = torch.finfo(q0.dtype).eps
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(eps)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(eps)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    if shortestpath:
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs()
    dot = dot.clamp(-1.0, 1.0)

    tangent = q1 - dot * q0
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    axis = torch.zeros_like(q0).scatter_(-1, q0.abs().argmin(dim=-1, keepdim=True), 1.0)
    fallback = axis - (axis * q0).sum(dim=-1, keepdim=True) * q0
    fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(eps)
    tangent = torch.where(tangent_norm > eps, tangent / tangent_norm.clamp_min(eps), fallback)
    angle = torch.acos(dot) + spin * torch.pi
    out = torch.cos(fraction * angle) * q0 + torch.sin(fraction * angle) * tangent
    if spin == 0:
        linear = (1.0 - fraction) * q0 + fraction * q1
        out = torch.where(dot > 0.9995, linear, out)
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(eps)
    out = torch.where(fraction == 0, q0, out)
    return torch.where(fraction == 1, q1, out)
