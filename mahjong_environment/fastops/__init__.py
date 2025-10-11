try:
    from .fastops import build_observation_i8_f32, compute_fast_actions
    FASTOPS_AVAILABLE = True
except Exception:
    FASTOPS_AVAILABLE = False

__all__ = [
    "FASTOPS_AVAILABLE",
    "build_observation_i8_f32",
    "compute_fast_actions",
]


