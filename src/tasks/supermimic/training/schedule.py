"""Cosine learning-rate decay in learning iterations."""

import math


def cosine_lr(step, decay_iterations, base_lr, final_ratio=0.03):
    progress = min(1.0, step / decay_iterations)
    return base_lr * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
