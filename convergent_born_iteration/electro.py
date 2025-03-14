"""
A module with functions to solve electro-magnetic problems.
"""
import scipy.constants as const
import jax
import jax.numpy as jnp
import jaxopt
import jaxopt.linear_solve

from convergent_born_iteration import log

log.getChild(__name__)

def precondition(grid_k, k0: float, permittivity, current_density):
    """
    Preconditions the electromagnetic problem.

    :param grid_k: A sequence with broadcastable ifftshifted k-space ranges.
    :param k0: The wavenumber in vacuum.
    :param permittivity: The relative permittivity of the material. Anisotropic materials have two extra axis on the left with shape (3, 3).
    :param current_density: The current density as a vector function of space. The polarization is the left-most axis.

    :return: The tuple (prec_forward, prec_y) with:
        # a function to compute the preconditioned forward problem
        # the preconditioned right hand side.
    """
    isotropic = permittivity.ndim < len(grid_k) + 2 or permittivity.shape[0] == 1
    log.info(f"Solving for an {'' if isotropic else 'an'}isotropic material...")

    log.debug('Defining some functions to work only on the polarization or only on the spatial dimensions...')
    if isotropic:
        @jax.jit
        def matrix_norm(_):
            return jnp.abs(_)
    else:
        @jax.jit
        def matrix_norm(_):
            return jnp.linalg.norm(_, axis=(0, 1))  # The latter can probably be faster by a custom implementation (see macromax)

    ft_kwargs = dict(axes=tuple(range(-len(grid_k), 0)), norm='ortho')

    @jax.jit
    def ft(x):
        """Fourier transform of the spatial dimensions."""
        return jnp.fft.fftn(x, **ft_kwargs)

    @jax.jit
    def ift(x):
        """Inverse Fourier transform of the spatial dimensions."""
        return jnp.fft.ifftn(x, **ft_kwargs)

    # The actual work is done here:
    log.debug('Scaling and shifting problem...')
    permittivity_bias = 1.25  # This can be chosen more optimally to minimize the following scaling factor, and maximize the convergence rate.
    scale = 1.1j * jnp.amax(matrix_norm(permittivity - permittivity_bias))  # Must be strictly larger than the norm in the polarization dimension
    # assert jnp.amax(jnp.abs((permittivity - permittivity_bias) / scale)) < 1, f'Incorrect scale.'

    y = -1j * k0 * const.c * const.mu_0 * current_density

    log.debug(f'Using permittivity bias {permittivity_bias}, and scaling the whole problem by {scale}.')

    k2 = sum((_ / k0) ** 2 for _ in grid_k)  # Use units so that k0 == 1

    @jax.jit
    def split_trans_long_ft(x_ft):
        """Split a k-space vector field into its transverse and longitudinal components."""
        dc = k2 == 0  # to avoid division by 0
        projection_coefficient = sum(k * c for k, c in zip(grid_k, x_ft)) / (k2 + dc)
        x_long_ft = sum(k * projection_coefficient for k in grid_k) * dc
        x_trans_ft = x_ft - x_long_ft
        return x_trans_ft, x_long_ft

    @jax.jit
    def shifted_approx_inv(y):
        """The inverse of the scaled and shifted-by-1 approximation to the scaled forward problem."""
        y_trans_ft, y_long_ft = split_trans_long_ft(ft(y))
        return ift(y_trans_ft / ((-k2 + permittivity_bias) / scale + 1) +
                   y_long_ft / (permittivity_bias / scale + 1)
                   )

    if isotropic:
        @jax.jit
        def shifted_discrepancy(x):
            """The discrepancy after approximation of the scaled isotropic problem, shifted by -1."""
            return ((permittivity - permittivity_bias) / scale - 1) * x
    else:
        @jax.jit
        def shifted_discrepancy(x):
            """The discrepancy after approximation of the scaled anisotropic problem, shifted by -1."""
            id = jnp.eye(3).reshape(3, 3, *([1] * len(grid_k)))  # The identity matrix in the polarization axes
            return jnp.einsum('ij...,j...,i...', (permittivity - (permittivity_bias / scale + 1) * id), x)


    @jax.jit
    def prec(x):
        """The preconditioner for accretive problem."""
        return -shifted_discrepancy(shifted_approx_inv(x / scale))

    @jax.jit
    def prec_forward(x):
        """The preconditioned problem does not actually require execution of the forward problem."""
        return -shifted_discrepancy(shifted_approx_inv(shifted_discrepancy(x)) + x)

    return prec_forward, prec(y)

def solve(grid_k, k0, permittivity, current_density, initial_E = None, implicit_diff: bool = True):
    """
    Solve for the electromagnetic field.

    :param grid_k: A tuple with the (ifftshifted) k_space grid, corresponding to the wavevectors after FFT.
    :param k0: The vacuum wavenumber.
    :param permittivity: The (relative) permittivity distribution can either have the spatial dimensions, or it can have
        a 3x3 matrix in the first (left-most) axes, for each point in space.
    :param current_density: The current density, with the first (left-most) axis the polarization vector, while the
        remaining axes are spatial dimensions.
    :param initial_E: An optional starting point for the solver.
    :param implicit_diff: Whether to compute implicit gradients during the fixed point iteration. Default: True.

    :return: The electromagnetic field, E, with the first (left-most) axis the polarization vector, while the remaining
        axes are spatial dimensions.
    """
    prec_forward, prec_y = precondition(grid_k, k0, permittivity, current_density)
    numerical_scale = jnp.amax(jnp.abs(prec_y))  # To avoid overflow or underflow with our machine precision
    prec_y /= numerical_scale
    x = prec_y if initial_E is None else initial_E / numerical_scale

    solver = jaxopt.FixedPointIteration(lambda _: prec_y - prec_forward(_) + _, maxiter=1000, tol=1e-3, jit=True, implicit_diff=implicit_diff)
    # bicgstab_solver = jax.jit(lambda: jaxopt.linear_solve.solve_bicgstab(prec_forward, prec_y, maxiter=1000, tol=1e-3))

    x = solver.run(x)[0]
    # x = bicgstab_solver()  # Takes about 4/3 the time as the fixed point run for simple problems.

    # Checking the intermediate, preconditioned, result...
    relative_residue_prec = jnp.linalg.norm(prec_forward(x) - prec_y) / jnp.linalg.norm(prec_y)
    log.info(f'Preconditioned relative residue of E: {relative_residue_prec:0.3e}.')

    return x * numerical_scale  # E
