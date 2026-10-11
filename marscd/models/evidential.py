"""v3  Evidential (subjective-logic) fusion of multi-view evidence for existence decisions.

For every T1 building j the stream accumulates OBSERVATION-WEIGHTED evidence that its footprint is
occupied at T2 (r_j) or empty (s_j):

    r_j += o_jt * softplus(l_jt),     s_j += o_jt * softplus(-l_jt),

where l_jt is the existence logit of the instance hypothesis anchored at j in frame t and o_jt the
z-buffer observability of j's footprint in that frame. With a uniform Beta(1,1) prior this yields the
binomial opinion (belief b, disbelief d, uncertainty u):

    b = r / (r + s + 2),   d = s / (r + s + 2),   u = 2 / (r + s + 2).

Unlike a probability, u separates "never really seen" from "seen and ambiguous": an unmatched T1 building
is reported as DEMOLISHED only when u < u_max, otherwise as UNOBSERVED. Training uses the evidential Beta
loss (expected cross-entropy + KL to the uniform Beta on misleading evidence, Sensoy et al. 2018).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def accumulate(r: Tensor, s: Tensor, idx: Tensor, logit: Tensor, obs: Tensor) -> tuple[Tensor, Tensor]:
    """Functional (autograd-friendly) evidence update for instances idx."""
    if idx.numel() == 0:
        return r, s
    r = r.index_add(0, idx, obs * F.softplus(logit))
    s = s.index_add(0, idx, obs * F.softplus(-logit))
    return r, s


def opinion(r: Tensor, s: Tensor, prior_w: float = 2.0) -> tuple[Tensor, Tensor, Tensor]:
    S = r + s + prior_w
    return r / S, s / S, prior_w / S


def _kl_beta_uniform(a: Tensor, b: Tensor) -> Tensor:
    """KL( Beta(a, b) || Beta(1, 1) )."""
    return (torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b)
            + (a - 1) * (torch.digamma(a) - torch.digamma(a + b))
            + (b - 1) * (torch.digamma(b) - torch.digamma(a + b)))


def edl_beta_loss(r: Tensor, s: Tensor, y: Tensor, mask: Tensor | None = None, kl_weight: float = 0.1) -> Tensor:
    """Evidential loss on accumulated evidence; y in {0, 1}; mask selects instances that received evidence."""
    a, b = r + 1.0, s + 1.0
    S = a + b
    nll = y * (torch.digamma(S) - torch.digamma(a)) + (1 - y) * (torch.digamma(S) - torch.digamma(b))
    a_t = y + (1 - y) * a  # remove evidence of the true class, penalise the misleading remainder
    b_t = (1 - y) + y * b
    loss = nll + kl_weight * _kl_beta_uniform(a_t, b_t)
    if mask is not None:
        loss = loss[mask]
    return loss.mean() if loss.numel() else r.sum() * 0


def novelty(W_new: Tensor, W_old: Tensor) -> Tensor:
    """Effective-sample-size discount for correlated video frames, measured on the prism texture atlas.

    novelty = sum_c dW_c / (1 + W_c) / sum_c dW_c  in (0, 1]: a frame that re-observes cells that are already
    well observed (consecutive 10 FPS frames) contributes little NEW evidence; a frame that sees new surface
    (a new facade, a new viewing angle) contributes fully. Without it, n correlated frames would be counted as
    n independent observations and the subjective-logic uncertainty would collapse too early.
    """
    tot = W_new.sum(-1)
    nov = (W_new / (1.0 + W_old)).sum(-1) / tot.clamp_min(1e-6)
    return torch.where(tot > 1e-6, nov, torch.zeros_like(nov))
