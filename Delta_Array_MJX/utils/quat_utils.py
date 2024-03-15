import jax.numpy as jnp
from jax import jit

@jit
def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z])

@jit
def quat_inner_product(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    product = w1*w2 + x1*x2 + y1*y2 + z1*z2
    return product

@jit
def orientation_cost(q1, q2):
    """Compute the angular difference between two quaternions in radians."""
    q_rel = quat_inner_product(quaternion_conjugate(q1), q2)
    angle = jnp.arccos(2 * q_rel**2 - 1)  # Clip to avoid numerical issues outside of [-1, 1]
    return angle
