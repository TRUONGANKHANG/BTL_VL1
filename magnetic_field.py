"""Midpoint Biot-Savart discretization of a circular current loop."""
import numpy as np

MU0 = 4 * np.pi * 1e-7  # conventional approximation, H/m


def create_segments(radius: float, n_segments: int):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Ban kinh R phai > 0.")
    if not isinstance(n_segments, (int, np.integer)) or n_segments < 3:
        raise ValueError("N phai la so nguyen >= 3.")
    dtheta = 2 * np.pi / n_segments
    theta = (np.arange(n_segments) + 0.5) * dtheta
    p = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(n_segments)))
    dl = np.column_stack((-radius * np.sin(theta), radius * np.cos(theta), np.zeros(n_segments))) * dtheta
    return p, dl


def field_at_points(points, radius: float, current: float, n_segments: int, chunk_size: int = 512):
    """Return (..., 3) field in Tesla. Mask points on the ideal wire with NaN.

    Positive current: counter-clockwise as viewed from +z.
    """
    if not np.isfinite(current):
        raise ValueError("Dong dien I phai huu han.")
    xyz = np.asarray(points, dtype=float)
    if xyz.shape[-1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("Toa do phai co 3 thanh phan huu han.")
    shape = xyz.shape[:-1]
    flat = xyz.reshape(-1, 3)
    segments, dl = create_segments(radius, n_segments)
    out = np.empty_like(flat)
    # A sample on the infinitely thin wire is singular. Exclude it exactly
    # (within floating-point-level tolerance), not a broad surrounding region.
    on_wire = (np.abs(np.hypot(flat[:, 0], flat[:, 1]) - radius) < 1e-12 * max(radius, 1)) & (np.abs(flat[:, 2]) < 1e-12 * max(radius, 1))
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        r = flat[start:stop, None, :] - segments[None, :, :]
        dist2 = np.sum(r*r, axis=2)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            denom = np.power(dist2, 1.5)
            dB = np.cross(dl[None, :, :], r) / denom[:, :, None]
            out[start:stop] = (MU0 * current / (4*np.pi)) * np.sum(dB, axis=1)
    out[on_wire] = np.nan
    return out.reshape(*shape, 3)


def axis_field(z, radius: float, current: float):
    z = np.asarray(z, dtype=float)
    return MU0 * current * radius**2 / (2 * (radius**2 + z**2)**1.5)


def center_field(radius: float, current: float):
    return MU0 * current / (2 * radius)
