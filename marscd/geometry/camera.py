"""Camera / pose utilities.

Conventions (used consistently across the code base):
  * World frame: local metric ENU (x = east, y = north, z = up). In practice the
    origin is the centre of the survey area in a projected CRS (e.g. CGCS2000
    3-degree Gauss-Kruger / UTM); SHP, DOM and raster-map grids live here.
  * Camera frame: OpenCV (x = right, y = down, z = forward).
  * Gimbal angles follow the DJI convention: yaw is the compass heading
    (clockwise from north), pitch is negative when looking down (-90 = nadir),
    roll about the optical axis. All angles are in radians.
  * A pose stores the world->camera transform X_c = R_cw X_w + t_cw.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _rx(a: Tensor) -> Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(*a.shape, 3, 3)


def _rz(a: Tensor) -> Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*a.shape, 3, 3)


def skew(w: Tensor) -> Tensor:
    """[..., 3] -> [..., 3, 3] cross-product matrix."""
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*w.shape[:-1], 3, 3)


def gimbal_to_R_wc(yaw: Tensor, pitch: Tensor, roll: Tensor) -> Tensor:
    """Camera-to-world rotation from DJI-style gimbal angles.

    R_wc = Rz(-yaw) @ R0 @ Rx(pitch) @ Rz(roll), where R0 maps a level camera
    looking north (x_c = east, y_c = down, z_c = north).
    """
    r0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                      dtype=yaw.dtype, device=yaw.device)
    return _rz(-yaw) @ r0 @ _rx(pitch) @ _rz(roll)


def se3_exp(xi: Tensor) -> tuple[Tensor, Tensor]:
    """Exponential map of se(3); xi = (omega, v) with shape [6]."""
    omega, v = xi[:3], xi[3:]
    theta = omega.norm().clamp_min(1e-12)
    W = skew(omega)
    W2 = W @ W
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    small = theta < 1e-4
    a = torch.where(small, 1.0 - theta**2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta**2 / 24.0, (1.0 - torch.cos(theta)) / theta**2)
    c = torch.where(small, 1.0 / 6.0 - theta**2 / 120.0, (theta - torch.sin(theta)) / theta**3)
    R = eye + a * W + b * W2
    V = eye + b * W + c * W2
    return R, V @ v


@dataclass
class Pose:
    """World->camera pose with intrinsics (single frame)."""

    R_cw: Tensor  # [3, 3]
    t_cw: Tensor  # [3]
    K: Tensor  # [3, 3]

    @property
    def R_wc(self) -> Tensor:
        return self.R_cw.transpose(-1, -2)

    @property
    def center(self) -> Tensor:
        return -(self.R_wc @ self.t_cw)

    @property
    def yaw(self) -> Tensor:
        """Compass heading of the optical axis projected onto the ground."""
        f = self.R_wc[:, 2]
        return torch.atan2(f[0], f[1])

    def to_camera(self, X_w: Tensor) -> Tensor:
        return X_w @ self.R_cw.transpose(-1, -2) + self.t_cw

    def left_update(self, xi: Tensor) -> "Pose":
        """T_cw <- exp(xi) * T_cw (perturbation expressed in the camera frame)."""
        dR, dt = se3_exp(xi)
        return Pose(dR @ self.R_cw, dR @ self.t_cw + dt, self.K)

    def detach(self) -> "Pose":
        return Pose(self.R_cw.detach(), self.t_cw.detach(), self.K.detach())

    @staticmethod
    def from_center(R_wc: Tensor, C: Tensor, K: Tensor) -> "Pose":
        R_cw = R_wc.transpose(-1, -2)
        return Pose(R_cw, -(R_cw @ C), K)


def pose_from_gnss(position: Tensor, yaw: Tensor, pitch: Tensor, roll: Tensor, K: Tensor) -> Pose:
    return Pose.from_center(gimbal_to_R_wc(yaw, pitch, roll), position, K)


def project(X_w: Tensor, pose: Pose, eps: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Project world points [..., 3] -> pixel coords [..., 2] and depth [...]."""
    X_c = pose.to_camera(X_w)
    z = X_c[..., 2]
    zc = z.clamp_min(eps)
    K = pose.K
    u = K[0, 0] * X_c[..., 0] / zc + K[0, 2]
    v = K[1, 1] * X_c[..., 1] / zc + K[1, 2]
    return torch.stack([u, v], -1), z


def projection_jacobian(X_c: Tensor, K: Tensor) -> Tensor:
    """d(u,v)/d(xi) for the left perturbation exp(xi) T_cw; X_c: [N,3] -> [N,2,6]."""
    x, y, z = X_c.unbind(-1)
    z = z.clamp_min(1e-3)
    fx, fy = K[0, 0], K[1, 1]
    zeros = torch.zeros_like(x)
    J_proj = torch.stack([
        torch.stack([fx / z, zeros, -fx * x / z**2], -1),
        torch.stack([zeros, fy / z, -fy * y / z**2], -1),
    ], -2)  # [N,2,3]
    J_pt = torch.cat([-skew(X_c), torch.eye(3, dtype=X_c.dtype, device=X_c.device).expand(X_c.shape[0], 3, 3)], -1)
    return J_proj @ J_pt  # [N,2,6]


def apply_world_rigid2d(pose: Pose, alpha: Tensor, center_xy: Tensor, delta_xy: Tensor) -> Pose:
    """Apply the planar world transform G(x) = Rz(alpha)(x - c) + c + delta to a camera.

    This is how a (x, y, yaw) correction estimated by BEV correlation is turned
    into a corrected 6-DoF pose (z, pitch and roll untouched).
    """
    Rz = _rz(alpha)
    C = pose.center
    c3 = torch.cat([center_xy, C.new_zeros(1)])
    d3 = torch.cat([delta_xy, C.new_zeros(1)])
    C_new = Rz @ (C - c3) + c3 + d3
    return Pose.from_center(Rz @ pose.R_wc, C_new, pose.K)


def ground_intersection(pose: Pose, uv: Tensor, ground_z: float | Tensor = 0.0) -> Tensor:
    """Intersect pixel rays [..., 2] with the plane z = ground_z; returns [..., 3]."""
    K_inv = torch.linalg.inv(pose.K)
    ones = torch.ones_like(uv[..., :1])
    rays_c = torch.cat([uv, ones], -1) @ K_inv.transpose(0, 1)
    rays_w = rays_c @ pose.R_wc.transpose(0, 1)
    C = pose.center
    s = (ground_z - C[2]) / rays_w[..., 2].clamp(max=-1e-6)
    return C + s.unsqueeze(-1) * rays_w
