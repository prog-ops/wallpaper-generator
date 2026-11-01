//! Lattice-artifact-free 2D Simplex Noise with exact analytic derivatives,
//! rotated multi-octave Curl Flow fields, and blue-noise-like Interleaved Gradient Dither.

use std::f32::consts::TAU;

const F2: f32 = 0.366_025_42; // (sqrt(3) - 1) / 2
const G2: f32 = 0.211_324_87; // (3 - sqrt(3)) / 6
const NORM_2D: f32 = 95.0;

/// 16 evenly distributed unit vectors around the circle to avoid axis bias.
const UNIT_GRADS: [(f32, f32); 16] = [
    (1.0, 0.0),
    (0.923_879_5, 0.382_683_43),
    (0.707_106_77, 0.707_106_77),
    (0.382_683_43, 0.923_879_5),
    (0.0, 1.0),
    (-0.382_683_43, 0.923_879_5),
    (-0.707_106_77, 0.707_106_77),
    (-0.923_879_5, 0.382_683_43),
    (-1.0, 0.0),
    (-0.923_879_5, -0.382_683_43),
    (-0.707_106_77, -0.707_106_77),
    (-0.382_683_43, -0.923_879_5),
    (0.0, -1.0),
    (0.382_683_43, -0.923_879_5),
    (0.707_106_77, -0.707_106_77),
    (0.923_879_5, -0.382_683_43),
];

#[inline]
fn hash_grad(ix: i32, iy: i32, seed: u64) -> (f32, f32) {
    let mut h = seed
        .wrapping_add((ix as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((iy as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    UNIT_GRADS[(h as usize) & 15]
}

/// Evaluates 2D Simplex noise in `[-1.0, 1.0]` along with its exact analytic
/// partial derivatives `(value, d_dx, d_dy)`.
#[inline]
pub fn simplex2d_grad(x: f32, y: f32, seed: u64) -> (f32, f32, f32) {
    let s = (x + y) * F2;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;

    let t = (i + j) as f32 * G2;
    let x0 = x - (i as f32 - t);
    let y0 = y - (j as f32 - t);

    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

    let x1 = x0 - i1 as f32 + G2;
    let y1 = y0 - j1 as f32 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    let mut val = 0.0_f32;
    let mut dx = 0.0_f32;
    let mut dy = 0.0_f32;

    let corners = [
        (x0, y0, i, j),
        (x1, y1, i + i1, j + j1),
        (x2, y2, i + 1, j + 1),
    ];

    for (xk, yk, ik, jk) in corners {
        let tk = 0.5 - xk * xk - yk * yk;
        if tk > 0.0 {
            let (gx, gy) = hash_grad(ik, jk, seed);
            let tk2 = tk * tk;
            let tk3 = tk2 * tk;
            let tk4 = tk2 * tk2;
            let dot = gx * xk + gy * yk;

            val += tk4 * dot;
            dx += tk4 * gx - 8.0 * tk3 * xk * dot;
            dy += tk4 * gy - 8.0 * tk3 * yk * dot;
        }
    }

    (val * NORM_2D, dx * NORM_2D, dy * NORM_2D)
}

/// Multi-octave rotated FBM returning `(value, d_dx, d_dy)` with exact chain-rule derivatives.
/// Octave rotation eliminates any residual triangular lattice alignment.
#[inline]
pub fn fbm2d_grad(
    mut x: f32,
    mut y: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    seed: u64,
) -> (f32, f32, f32) {
    // Golden-ratio-inspired octave rotation (~38.7 deg)
    const COS_A: f32 = 0.780_430_4;
    const SIN_A: f32 = 0.625_242_7;

    let mut total = 0.0_f32;
    let mut total_dx = 0.0_f32;
    let mut total_dy = 0.0_f32;

    let mut amp = 1.0_f32;
    let mut amp_sum = 0.0_f32;

    // Track cumulative Jacobian J = d(u, v)/d(x0, y0)
    let mut j00 = 1.0_f32;
    let mut j01 = 0.0_f32;
    let mut j10 = 0.0_f32;
    let mut j11 = 1.0_f32;

    for oct in 0..octaves {
        let oct_seed = seed.wrapping_add((oct as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
        let (v, du, dv) = simplex2d_grad(x, y, oct_seed);

        total += amp * v;
        total_dx += amp * (du * j00 + dv * j10);
        total_dy += amp * (du * j01 + dv * j11);
        amp_sum += amp;

        amp *= gain;

        // Next octave coordinates: (x', y') = lacunarity * R * (x, y)
        let nx = lacunarity * (COS_A * x - SIN_A * y);
        let ny = lacunarity * (SIN_A * x + COS_A * y);
        x = nx;
        y = ny;

        let nj00 = lacunarity * (COS_A * j00 - SIN_A * j10);
        let nj01 = lacunarity * (COS_A * j01 - SIN_A * j11);
        let nj10 = lacunarity * (SIN_A * j00 + COS_A * j10);
        let nj11 = lacunarity * (SIN_A * j01 + COS_A * j11);
        j00 = nj00;
        j01 = nj01;
        j10 = nj10;
        j11 = nj11;
    }

    let inv = if amp_sum > 0.0 { 1.0 / amp_sum } else { 1.0 };
    (total * inv, total_dx * inv, total_dy * inv)
}

/// Divergence-free 2D Curl Velocity `(vx, vy) = (dPsi/dy, -dPsi/dx)`.
#[inline]
pub fn curl_velocity(
    x: f32,
    y: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    seed: u64,
) -> (f32, f32) {
    let (_psi, dpsi_dx, dpsi_dy) = fbm2d_grad(x, y, octaves, lacunarity, gain, seed);
    (dpsi_dy, -dpsi_dx)
}

/// High-frequency blue-noise-like Triangular PDF dither in `[-0.85, 0.85]`
/// using two phase-shifted Interleaved Gradient Noise taps.
#[inline]
pub fn blue_tpdf_dither(px: u32, py: u32, seed: u32) -> f32 {
    let fx = px as f32 + (seed & 1023) as f32 * 1.618_034;
    let fy = py as f32 + ((seed >> 10) & 1023) as f32 * 2.414_213_7;

    let ign1 = (52.982_918 * (0.067_110_56 * fx + 0.005_837_15 * fy).fract()).fract();
    let ign2 =
        (52.982_918 * (0.067_110_56 * (fx + 17.0) + 0.005_837_15 * (fy + 29.0)).fract()).fract();

    // Triangular distribution centered at 0.0
    let _ = TAU; // keep const import clean if unused elsewhere
    (ign1 - ign2) * 0.85
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let eps = 1e-3_f32;
        let seed = 0xCAFE_BABE_1234_5678;

        for step in 0..25 {
            let x = -2.3 + step as f32 * 0.193;
            let y = 1.7 - step as f32 * 0.147;

            let (_v, dx, dy) = fbm2d_grad(x, y, 3, 2.05, 0.5, seed);
            let (vx_p, _, _) = fbm2d_grad(x + eps, y, 3, 2.05, 0.5, seed);
            let (vx_m, _, _) = fbm2d_grad(x - eps, y, 3, 2.05, 0.5, seed);
            let (vy_p, _, _) = fbm2d_grad(x, y + eps, 3, 2.05, 0.5, seed);
            let (vy_m, _, _) = fbm2d_grad(x, y - eps, 3, 2.05, 0.5, seed);

            let fd_x = (vx_p - vx_m) / (2.0 * eps);
            let fd_y = (vy_p - vy_m) / (2.0 * eps);

            assert!(
                (dx - fd_x).abs() < 2.5e-2 && (dy - fd_y).abs() < 2.5e-2,
                "Gradient mismatch at ({x}, {y}): analytic=({dx}, {dy}), fd=({fd_x}, {fd_y})"
            );
        }
    }
}

