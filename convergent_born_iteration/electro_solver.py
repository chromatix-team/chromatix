"""
A module with functions to solve electro-magnetic problems.

See example_solve2d.py for an example.
"""
import jax
import jax.numpy as jnp
import jaxopt
import scipy.constants as const

from chromatix.utils import dim, Grid


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
    def shifted_norm(_, s):
        """Computes the norm of the shifted (block-)diagonal potential."""
        return jnp.amax(jnp.linalg.norm(jnp.moveaxis(_, (0, 1), (-2, -1)) - s * jnp.eye(_.shape[0]), axis=(-2, -1)))

    f = lambda _: shifted_norm(permittivity, _) ** 2
    df = jax.grad(f)
    x = 1.6 + 0.25j
    # x = 1.0 + 0.05j
    fx = f(x)
    steps = 0.75 ** jnp.arange(10)
    for it in range(10):
        steps = steps * 0.5
        dfx = df(x)
        fxs = jnp.array([f(x - _ * dfx) for _ in steps])
        step_index = jnp.argmin(fxs)
        improved = jnp.any(fxs[step_index] < fx)
        x = x - steps[step_index] * dfx * improved
        fx = jnp.minimum(fx, fxs[step_index])
        # jax.debug.print('f({x}) = {fx}  dfx={dfx}, fxs={fxs}', x=x, fx=fx, dfx=dfx, fxs=fxs)

    value = fx
    shift = x

    scale = 1.1j * value ** 0.5

    return shift, scale

def precondition(grid: Grid, k0: float, permittivity, current_density, adjoint: bool = False):
    """
    Preconditions the electromagnetic problem.

    :param grid: The uniform plaid spatial sampling grid for this calculation.
    :param k0: The vacuum wavenumber.
    :param permittivity: The relative permittivity of the material. Anisotropic materials have two extra axis on the left with shape (3, 3).
    :param current_density: The current density as a vector function of space. The polarization is the left-most axis.
    :param adjoint: Precondition for the adjoint problem instead. Default: False.

    :return: The tuple (prec_forward, prec_y) with:

        1. A callable function to compute the preconditioned forward problem for an electric field distribution, and

        2. The preconditioned right hand side.

    """
    grid_shape = current_density.shape[1:]  # FIXME: for some reason we cannot use grid.shape when JIT-ing.
    grid_k = [dim.to_axis(jnp.fft.ifftshift(jnp.arange(sh) - (sh // 2)) * st * 2 * jnp.pi / k0,
                          axis=-len(grid_shape) + _
                          )
              for _, (sh, st) in enumerate(zip(grid_shape, 1 / jnp.array(grid_shape) / grid.step))]  # Use units so that k0 == 1.

    permittivity_bias, scale = get_shift_and_scale(permittivity)
    scale_inv = 1 / (scale * (1 - 2 * adjoint))

    subscripts = 'ij...,...->...' if permittivity.shape[0] == 1 else 'ij...,j...->i...'
    scaled_and_shifted_permittivity_bias = permittivity_bias * scale_inv + 1
    scaled_permittivity = permittivity * scale_inv
    del permittivity

    def shifted_discrepancy(x):
        """The discrepancy after approximation of the scaled (an)isotropic problem, shifted by -1."""
        return jnp.einsum(subscripts, scaled_permittivity, x) - scaled_and_shifted_permittivity_bias * x

    def split_trans_long_ft(y_ft):
        """Split a k-space vector field into its transverse and longitudinal components."""
        k2 = sum(_ ** 2 for _ in grid_k)  # This is more memory & computationally efficient when computed on-the-fly every time
        dc = k2 == 0  # just to avoid division by 0
        projection_coefficient_div_norm_k = sum(k * y_ft_c for k, y_ft_c in zip(grid_k, y_ft)) / (k2 + dc)  # dot-product with over-normalized k-vector
        grid_k_3d = (*grid_k, *([0] * (y_ft.shape[0] - len(grid_k))))  # 0-pad sequence of vectors
        y_long_ft = jnp.stack([k * projection_coefficient_div_norm_k for k in grid_k_3d])
        y_trans_ft = y_ft - y_long_ft
        return y_trans_ft, y_long_ft

    def shifted_approx_inv(y):
        """The inverse of the scaled and shifted-by-1 approximation to the scaled forward problem."""
        k2 = sum(_ ** 2 for _ in grid_k)  # This is more memory & computationally efficient when computed on-the-fly every time
        ft_kwargs = dict(axes=tuple(range(-len(grid_k), 0)))  # use , norm='ortho' to avoid numerical problems with low numerical precision.

        y_ft = jnp.fft.fftn(y, **ft_kwargs)
        y_trans_ft, y_long_ft = split_trans_long_ft(y_ft)
        return jnp.fft.ifftn(y_trans_ft / (-k2 / scale + scaled_and_shifted_permittivity_bias) +
                             y_long_ft / scaled_and_shifted_permittivity_bias,
                             **ft_kwargs
                             )

    def prec(y):
        """The preconditioner for accretive problem."""
        return -shifted_discrepancy(shifted_approx_inv(y / scale))

    def prec_forward(x):
        """The preconditioned problem does not actually require execution of the forward problem."""
        return -shifted_discrepancy(shifted_approx_inv(shifted_discrepancy(x)) + x)

    return prec_forward, prec(-1j * k0 * const.c * const.mu_0 * current_density)

def solve(grid: Grid, k0: float, permittivity, current_density, initial_E = None,
          maxiter: int = 1000, tol: float = 1e-3,
          adjoint: bool = False,
          implicit_diff: bool = True,
          ):
    """
    Solve for the electromagnetic field.

    :param grid: The uniform plaid spatial sampling grid for this calculation.
    :param k0: The vacuum wavenumber.
    :param permittivity: The (relative) permittivity distribution can either have the spatial dimensions, or it can have
        a 3x3 matrix in the first (left-most) axes, for each point in space.
    :param current_density: The current density, with the first (left-most) axis the polarization vector, while the
        remaining axes are spatial dimensions.
    :param initial_E: An optional starting point for the solver.
    :param maxiter: The maximum number of iterations.
    :param tol: The tolerance for the convergence criterion.
    :param adjoint: Invert the adjoint problem instead. Default: False.
    :param implicit_diff: Whether to compute implicit gradients during the fixed point iteration. Default: True.

    :return: The electromagnetic field, E, with the first (left-most) axis the polarization vector, while the remaining
        axes are spatial dimensions.
    """
    while permittivity.ndim < 2 + len(grid.shape):
        permittivity = permittivity[jnp.newaxis]  # Add singleton dimensions on the left.
    if adjoint:
        permittivity = permittivity.transpose(0, 1).conj()
    prec_forward, prec_y = precondition(grid, k0, permittivity, current_density, adjoint=adjoint)
    numerical_scale = jnp.amax(jnp.abs(prec_y))  # To avoid overflow or underflow with our machine precision
    prec_y /= numerical_scale
    x = prec_y if initial_E is None else initial_E / numerical_scale

    solver = jaxopt.FixedPointIteration(lambda _: prec_y - prec_forward(_) + _,
                                        maxiter=maxiter, tol=tol, implicit_diff=implicit_diff,
                                        )
    x, optim_state = solver.run(x)

    # bicgstab_solver = jaxopt.linear_solve.solve_bicgstab(prec_forward, prec_y, maxiter=maxiter, tol=tol)
    # x = bicgstab_solver()  # Takes about 33% more than the fixed point run for simple problems.

    return x * numerical_scale  # E
