import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as spa
import jaxopt
from optimistix import NonlinearCG, minimise

from chromatix.experimental.modified_born_series.sample import Sample, Source


def get_shift_and_scale(permittivity):
    """
    Determine an appropriate shift and scale, so that:

        * max |permittivity - shift| is small,

        * max |permittivity - shift| < |scale|, and

        * forward / scale is accretive (non-symmetric positive definite).

    Ideally, the shift should be optimized to minimize the scale,
    as that is approximately inversely proportional to the convergence rate.

    :param permittivity: The relative permittivity of the material. Anisotropic materials have two extra axis on the left with shape (3, 3).
    :return: The complex tuple (shift, scale)
    """

    def loss_fn(params, _):
        """Computes the norm of the shifted (block-)diagonal potential."""
        # NOTE: This needs to be a norm over n axes for tensorial
        shift = params[0] + 1j * params[1]
        return jnp.max(jnp.abs(permittivity - shift)) ** 2

    solver = NonlinearCG(rtol=1e-3, atol=1e-3)
    solution = minimise(loss_fn, solver, jnp.array([1.6, 0.25]))

    # Parsing output
    shift = solution.value[0] + 1j * solution.value[1]
    scale = 1.1j * jnp.sqrt(solution.state.f_info.f)
    return shift, 1 / scale


def precondition(sample: Sample, source: Source):
    """
    Preconditions the electromagnetic problem.

    :param grid: The uniform plaid spatial sampling grid for this calculation.
    :param k0: The vacuum wavenumber.
    :param permittivity: The relative permittivity of the material. Anisotropic materials have two extra axis on the left with shape (3, 3).
    :param source: Proportional to the current density: source = -1j / k0 * const.c * const.mu_0 * current_density

    :return: The tuple (prec_forward, prec_y) with:

        1. A callable function to compute the preconditioned forward problem for an electric field distribution, and

        2. The preconditioned right hand side.

    """

    # Normalise the grid on k0
    k_grid = sample.k_grid / source.k0
    k_sq = jnp.sum(k_grid**2, axis=-1, keepdims=True)

    # Result should be independent of preconditioner, so we don't calculate the grads.
    bias, scale = get_shift_and_scale(jax.lax.stop_gradient(sample.permittivity))
    k_bias_sq = bias * scale + 1

    def shifted_discrepancy(x):
        """The discrepancy after approximation of the scaled (an)isotropic problem, shifted by -1."""
        # NOTE: We only do scalar permittivity; this will need to be a tensordot when doing matrices
        return x * (scale * sample.permittivity[..., None] - k_bias_sq)

    def split_trans_long_ft(y_ft):
        """Split a k-space vector field into its transverse and longitudinal components."""
        projection_coefficient = jnp.sum(k_grid * y_ft, axis=-1, keepdims=True)
        y_long_ft = jnp.where(k_sq != 0, k_grid / k_sq * projection_coefficient, 0.0)
        y_trans_ft = y_ft - y_long_ft

        return y_trans_ft, y_long_ft

    def shifted_approx_inv(y):
        """The inverse of the scaled and shifted-by-1 approximation to the scaled forward problem."""
        y_ft = jnp.fft.fftn(y, axes=(0, 1, 2))
        y_trans_ft, y_long_ft = split_trans_long_ft(y_ft)
        y_inv = y_trans_ft / (k_bias_sq - scale * k_sq) + y_long_ft / k_bias_sq
        return jnp.fft.ifftn(y_inv, axes=(0, 1, 2))

    def prec(y):
        """The preconditioner for accretive problem."""
        return -shifted_discrepancy(shifted_approx_inv(y * scale))

    def prec_forward(x):
        """The preconditioned problem does not actually require execution of the forward problem."""
        return -shifted_discrepancy(shifted_approx_inv(shifted_discrepancy(x)) + x)

    # For numerical reasons
    prec_y = prec(source.field)
    num_scale = jnp.max(jnp.abs(prec_y))  # for numerical reasons
    return prec_forward, prec_y / num_scale, num_scale


def solve(
    sample: Sample,
    source: Source,
    initial_E=None,
    maxiter: int = 1000,
    tol: float = 1e-3,
    implicit_diff: bool = True,
    use_bicgstab: bool = False,
):
    """
    Solve for the electromagnetic field.

    :param grid: The uniform plaid spatial sampling grid for this calculation.
    :param k0: The vacuum wavenumber.
    :param permittivity: The (relative) permittivity distribution can either have the spatial dimensions, or it can have
        a 3x3 matrix in the first (left-most) axes, for each point in space.
    :param current_density: The current density, with the first (left-most) axis the polarization vector, while the
        remaining axes are spatial dimensions.
    :param source: Pre-scaled alternative to current_density: source = -1j / k0 * const.c * const.mu_0 * current_density
    :param initial_E: An optional starting point for the solver.
    :param maxiter: The maximum number of iterations.
    :param tol: The tolerance for the convergence criterion.
    :param implicit_diff: Whether to compute implicit gradients during the fixed point iteration. Default: True.
    :param use_bicgstab: By default (False), the memory-efficient fixed-point iteration is used. When set to True,
        the faster-converging stabilized bi-conjugate-gradient algorithm (BiCGStab) is used instead.

    :return: The electromagnetic field, E, with the first (left-most) axis the polarization vector, while the remaining
        axes are spatial dimensions.
    """

    prec_forward, prec_y, scale = precondition(sample, source)
    x = prec_y if initial_E is None else initial_E / scale

    if not use_bicgstab:
        # jaxopt.AndersonAcceleration generally converges faster than jaxopt.FixedPointIteration but it uses more memory and triggers complex cast warnings
        solver = jaxopt.FixedPointIteration(
            lambda _: prec_y - prec_forward(_) + _,
            maxiter=maxiter,
            tol=tol,
            implicit_diff=implicit_diff,
        )
        x, optim_state = solver.run(x)
    else:
        x, optim_state = spa.bicgstab(prec_forward, prec_y, x, maxiter=maxiter, tol=tol)

    return x * scale  # E
