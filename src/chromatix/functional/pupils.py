from ..field import Field

__all__ = ["circular_pupil", "square_pupil"]


def circular_pupil(field: Field, w: float) -> Field:
    mask = field.l2_sq_grid <= ((field.spectrum[..., 0] / field.spectrum) * w / 2) ** 2
    return field * mask


def square_pupil(field: Field, w: float) -> Field:
    mask = field.linf_grid <= (field.spectrum[..., 0] / field.spectrum) * w / 2
    return field * mask
