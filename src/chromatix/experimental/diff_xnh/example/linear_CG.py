from typing import Callable
from optimistix import Solution, RESULTS
import jax
from jaxtyping import Array
import equinox as eqx
import jax.numpy as jnp


# This little trick only works if we input a complex field, and take the grad
class CGState(eqx.Module):
    descent: Array
    grad: Array
    first_step: bool


def descent_fn(state, grad):
    # Finds the descent direction
    def initial_update_descent(state, grad):
        descent = jax.tree.map(lambda x: -x, grad)
        return CGState(descent, grad, False)

    def update_descent(state, grad):
        def dai_yuan(descent, grad, grad_prev):
            beta = jnp.sum(grad**2) / jnp.sum(descent * (grad - grad_prev))
            return -grad + beta * descent

        descent = jax.tree.map(dai_yuan, state.descent, grad, state.grad)
        return CGState(descent, grad, False)

    return jax.lax.cond(
        state.first_step, initial_update_descent, update_descent, state, grad
    )


def line_search(norm_fn, approx, forward_grad, decrease_factor, step_init):
    # Finds the optimal step_size
    def body_fn(carry):
        (step_size, idx, _) = carry
        estimate = jax.tree.map(lambda x, y: x + step_size * y, approx, forward_grad)
        diff = norm_fn(approx) - norm_fn(estimate)
        return (step_size * decrease_factor, idx + 1, diff)

    def cond_fn(carry):
        (step_size, _, diff) = carry
        return jnp.logical_and(diff < 0, step_size > 1 / 64)

    step_size, _, _ = jax.lax.while_loop(cond_fn, body_fn, (step_init, 0, -1.0))
    return step_size / decrease_factor


@eqx.filter_jit
def linear_cg(
    forward_fn: Callable,
    norm_fn: Callable,
    params: tuple,
    args: tuple = (),
    *,
    max_steps: int = 20,
    decrease_factor: float = 0.5,
    step_init: float = 0.5,
    state: CGState | None = None,
):
    def loss_fn(params, args):
        approx = forward_fn(params, args)
        return norm_fn(approx, args), approx

    def cg_step(step_idx, carry):
        params, state, losses = carry
        # Calculate the graident
        (loss, approx), grad = jax.value_and_grad(loss_fn, has_aux=True)(params, args)
        grad = jax.tree.map(
            lambda x: jnp.conj(x), grad
        )  # NOTE: with complex differentiation we need to take the conjugate

        # Get descent step
        state = descent_fn(state, grad)

        # Line search
        forward_grad = forward_fn(state.descent, args)
        step_size = line_search(
            lambda x: norm_fn(x, args), approx, forward_grad, decrease_factor, step_init
        )

        # Update parameters
        params = jax.tree.map(
            lambda x, descent: x + step_size * descent, params, state.descent
        )
        losses = losses.at[step_idx].set(loss)
        return (params, state, losses)

    if state is None:
        state = CGState(
            jax.tree.map(jnp.zeros_like, params),
            jax.tree.map(jnp.zeros_like, params),
            True,
        )
    params, state, losses = jax.lax.fori_loop(
        0, max_steps, cg_step, (params, state, jnp.zeros((max_steps,)))
    )

    return Solution(
        value=params,
        result=RESULTS.successful,
        aux=None,
        stats={"loss": losses},
        state=state,
    )
