import jax
import jax.numpy as jnp

@jax.jit
def quat_normalize(q: jax.Array) -> jax.Array:
    return q / jnp.linalg.norm(q)

@jax.jit
def quat_conjugate(q: jax.Array) -> jax.Array:
    return q * jnp.array([1., -1., -1., -1.])

@jax.jit
def quat_multiply(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return jnp.stack([w, x, y, z], axis=-1)

@jax.jit
def quat_apply(q: jax.Array, points: jax.Array) -> jax.Array:
    # Represent points as pure quaternions (w=0)
    points_q = jnp.pad(points, ((0, 0), (1, 0)))

    # Rotate using the formula: p' = q * p * q_conjugate
    q_conj = quat_conjugate(q)
    rotated_points_q = quat_multiply(quat_multiply(q, points_q), q_conj)
    
    # Extract the vector part of the resulting quaternions
    return rotated_points_q[..., 1:]

@jax.jit
def quat_from_euler_z(angle: float) -> jax.Array:
    half_angle = angle / 2.0
    w = jnp.cos(half_angle)
    z = jnp.sin(half_angle)
    # Returns [w, x, y, z]
    return jnp.array([w, 0.0, 0.0, z])

@jax.jit
def quat_from_quat(q: jax.Array, scalar_first: bool = True) -> jax.Array:
    if not scalar_first:
        # If input is [x, y, z, w], convert to [w, x, y, z]
        q = jnp.roll(q, shift=1)
    
    return quat_normalize(q)