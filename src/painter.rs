//! Generative Painterly Gradient Engine.
//!
//! Synthesizes non-classic, velvety, fine-blended abstract gradient wallpapers by combining:
//! 1. Smooth low-frequency divergence-free Curl Flow + sweeping off-axis fluid currents
//!    (organic wet-on-wet curvature with zero high-frequency crinkling or singularities).
//! 2. $C^\infty$-smooth Curved Cubic-Bézier Brush Ribbons with continuous asymmetric feathering
//!    (lost-and-found painterly edges) and tapered brush pressure.
//! 3. Local inverse-power manifold blending $(D_i + \delta(x,y))^{-p(x,y)}$ so adjacent colors
//!    melt richly into each other via 16-band Kubelka-Munk + Oklab pigment physics without
//!    contaminating distant regions into mud.

use std::f32::consts::{PI, TAU};

use rand::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::color::{PreparedPigment, Srgb8, mix_pigments_painterly};
use crate::noise::{blue_tpdf_dither, curl_velocity, fbm2d_grad, simplex2d_grad};

const SEGMENTS_PER_RIBBON: usize = 12;

/// Rendered RGBA8 wallpaper buffer + metadata about the random painterly composition.
#[derive(Clone)]
pub struct GeneratedWallpaper {
    pub width: u32,
    pub height: u32,
    /// Tightly packed `RGBA8` pixels (`width * height * 4` bytes).
    pub rgba: Vec<u8>,
    pub seed: u64,
    pub style_label: &'static str,
}

#[derive(Clone, Copy)]
struct SweepCurrent {
    cx: f32,
    cy: f32,
    inv_radius2: f32,
    swirl: f32,
    push_x: f32,
    push_y: f32,
}

#[derive(Clone, Copy)]
struct RibbonSegment {
    ax: f32,
    ay: f32,
    abx: f32,
    aby: f32,
    inv_len2: f32,
    nx: f32,
    ny: f32,
    core_width: f32,
    scale_pos: f32,
    scale_neg: f32,
}

#[derive(Clone)]
struct BrushRibbon {
    color_idx: usize,
    presence: f32,
    segments: [RibbonSegment; SEGMENTS_PER_RIBBON],
}

/// Complete randomized generative plan for a single painterly canvas.
pub struct PainterlyPlan {
    pub seed: u64,
    pub style_label: &'static str,
    pigments: Vec<PreparedPigment>,
    currents: Vec<SweepCurrent>,
    ribbons: Vec<BrushRibbon>,

    // Gentle, velvety fluid-flow warp parameters (scaled for derivative norm ~3.5)
    macro_freq: f32,
    macro_advect: f32,
    mid_freq: f32,
    mid_advect: f32,

    // Spatially varying saduran (blending softness & power)
    delta_base: f32,
    delta_var: f32,
    power_base: f32,
    power_var: f32,
    reach_decay: f32,
    saduran_freq: f32,
    glaze_bias: f32,
    atm_veil_amp: f32,
}

impl PainterlyPlan {
    /// Creates a brand-new completely random painterly composition and blending plan
    /// for the given `2..=7` colors.
    pub fn random_from_palette(colors: &[Srgb8]) -> Self {
        let mut os_rng = rand::rng();
        let seed: u64 = os_rng.random();
        Self::from_palette_with_seed(colors, seed)
    }

    pub fn from_palette_with_seed(colors: &[Srgb8], seed: u64) -> Self {
        assert!(
            (2..=7).contains(&colors.len()),
            "Palette must contain between 2 and 7 colors"
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let pigments: Vec<PreparedPigment> = colors
            .iter()
            .copied()
            .map(PreparedPigment::from_srgb8)
            .collect();
        let n = pigments.len();

        // 5 distinct painterly saduran (blending) moods, randomized continuously per seed.
        // Note: curl_velocity derivative norm is ~3.5, so advect ~ 0.04..0.11 produces
        // a graceful 15%..38% canvas-height organic sweep without crinkling!
        let archetype: u32 = rng.random_range(0..5);
        let (
            style_label,
            macro_freq,
            macro_advect,
            mid_freq,
            mid_advect,
            delta_base,
            delta_var,
            power_base,
            power_var,
            glaze_bias,
        ) = match archetype {
            0 => (
                "Sfumato Atmospheric",
                rng.random_range(0.32..0.52),
                rng.random_range(0.055..0.090),
                rng.random_range(0.75..1.10),
                rng.random_range(0.016..0.030),
                rng.random_range(0.14..0.21),
                rng.random_range(0.05..0.09),
                rng.random_range(2.35..2.95),
                rng.random_range(0.45..0.75),
                rng.random_range(-0.20..0.45),
            ),
            1 => (
                "Silk Stain Wash",
                rng.random_range(0.42..0.68),
                rng.random_range(0.075..0.115),
                rng.random_range(0.90..1.35),
                rng.random_range(0.022..0.038),
                rng.random_range(0.09..0.15),
                rng.random_range(0.05..0.08),
                rng.random_range(2.85..3.55),
                rng.random_range(0.65..1.05),
                rng.random_range(0.10..0.70),
            ),
            2 => (
                "Velvet Glaze Sweep",
                rng.random_range(0.28..0.46),
                rng.random_range(0.065..0.100),
                rng.random_range(0.70..1.05),
                rng.random_range(0.014..0.026),
                rng.random_range(0.11..0.17),
                rng.random_range(0.06..0.10),
                rng.random_range(2.60..3.30),
                rng.random_range(0.60..0.95),
                rng.random_range(0.25..0.80),
            ),
            3 => (
                "Turner Twilight Veil",
                rng.random_range(0.36..0.58),
                rng.random_range(0.080..0.120),
                rng.random_range(0.85..1.25),
                rng.random_range(0.018..0.034),
                rng.random_range(0.13..0.20),
                rng.random_range(0.05..0.09),
                rng.random_range(2.45..3.10),
                rng.random_range(0.55..0.85),
                rng.random_range(-0.35..0.35),
            ),
            _ => (
                "Soft Impasto Flow",
                rng.random_range(0.45..0.72),
                rng.random_range(0.070..0.108),
                rng.random_range(1.00..1.45),
                rng.random_range(0.024..0.042),
                rng.random_range(0.08..0.13),
                rng.random_range(0.04..0.08),
                rng.random_range(3.05..3.85),
                rng.random_range(0.75..1.15),
                rng.random_range(0.00..0.65),
            ),
        };

        // Scale blend radius and ribbon spans smoothly with palette size N (2..=7)
        // so 6-7 color canvases keep every pigment jewel-clear without overlapping mud.
        let n_scale = (3.0 / (n as f32 + 1.0)).sqrt(); // 1.0 at N=2, 0.77 at N=4, 0.61 at N=7
        let delta_base = delta_base * n_scale;
        let delta_var = delta_var * n_scale;
        let power_base = power_base + (n.saturating_sub(3) as f32) * 0.14;
        let reach_decay = 2.0 + 0.68 * (n as f32);

        let saduran_freq = rng.random_range(0.55..0.95);
        let atm_veil_amp = rng.random_range(0.010..0.022);

        // 1. Broad, graceful off-axis painterly currents (gentle S-bends & swirls)
        let num_currents = rng.random_range(2..=3);
        let mut currents = Vec::with_capacity(num_currents);
        for _ in 0..num_currents {
            let angle = rng.random_range(0.0..TAU);
            let push_mag = rng.random_range(0.10..0.24);
            let radius = rng.random_range(0.65..1.25);
            currents.push(SweepCurrent {
                cx: rng.random_range(-0.75..0.75),
                cy: rng.random_range(-0.48..0.48),
                inv_radius2: 1.0 / (radius * radius),
                swirl: rng.random_range(-0.28..0.28),
                push_x: angle.cos() * push_mag,
                push_y: angle.sin() * push_mag,
            });
        }

        let mut plan = Self {
            seed,
            style_label,
            pigments,
            currents,
            ribbons: Vec::new(),
            macro_freq,
            macro_advect,
            mid_freq,
            mid_advect,
            delta_base,
            delta_var,
            power_base,
            power_var,
            reach_decay,
            saduran_freq,
            glaze_bias,
            atm_veil_amp,
        };

        // 2. Shuffle color roles randomly on every generation
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        // Choose a random overarching compositional flow direction for this painting,
        // with organic variation per stroke so strokes stay cohesive without colliding into X-cross mud
        let composition_flow_angle = rng.random_range(0.0..TAU);

        // Generate maximally separated asymmetric screen anchors across the 16:9 canvas
        let anchors = generate_organic_anchors(n, &mut rng);

        for (slot, &color_idx) in order.iter().enumerate() {
            let (sx, sy) = anchors[slot];
            let (wx, wy) = plan.warp_point(sx, sy);

            let angle_jitter = if n <= 3 && slot == n - 1 {
                rng.random_range(-1.05..1.05)
            } else {
                rng.random_range(-0.58..0.58)
            };
            let stroke_angle = composition_flow_angle + angle_jitter;

            let length = rng.random_range(0.95..1.68) * n_scale;
            let core_belly = rng.random_range(0.075..0.165) * n_scale;
            let presence = rng.random_range(0.92..1.28);

            plan.ribbons.push(build_curved_ribbon(
                color_idx,
                wx,
                wy,
                stroke_angle,
                length,
                core_belly,
                presence,
                &mut rng,
            ));
        }

        // 3. Secondary echo / glaze passages so colors weave organically across the canvas
        let echo_count = match n {
            2 => rng.random_range(2..=3),
            3 | 4 => rng.random_range(2..=3),
            _ => rng.random_range(1..=2),
        };

        for k in 0..echo_count {
            let color_idx = order[k % n];
            let sx = rng.random_range(-0.76..0.76);
            let sy = rng.random_range(-0.44..0.44);
            let (wx, wy) = plan.warp_point(sx, sy);

            let stroke_angle = composition_flow_angle + rng.random_range(-0.85..0.85);
            let length = rng.random_range(0.65..1.25) * n_scale;
            let core_belly = rng.random_range(0.04..0.09) * n_scale;
            let presence = rng.random_range(0.38..0.68);

            plan.ribbons.push(build_curved_ribbon(
                color_idx,
                wx,
                wy,
                stroke_angle,
                length,
                core_belly,
                presence,
                &mut rng,
            ));
        }

        plan
    }

    /// Smoothly warps aspect-normalized coordinate `(x, y)` through gentle off-axis currents
    /// and 2-stage divergence-free Curl Flow. Zero discontinuities or high-frequency crinkling.
    #[inline]
    fn warp_point(&self, mut x: f32, mut y: f32) -> (f32, f32) {
        for c in &self.currents {
            let dx = x - c.cx;
            let dy = y - c.cy;
            let falloff = (-(dx * dx + dy * dy) * c.inv_radius2).exp();
            x += (-dy * c.swirl + c.push_x) * falloff;
            y += (dx * c.swirl + c.push_y) * falloff;
        }

        // Macro curl streamline sweep (midpoint integration)
        let s1 = self.seed ^ 0xA5A5_5A5A_F0F0_0F0F;
        let (v1x, v1y) =
            curl_velocity(x * self.macro_freq, y * self.macro_freq, 2, 1.95, 0.45, s1);
        let mx = x + 0.5 * self.macro_advect * v1x;
        let my = y + 0.5 * self.macro_advect * v1y;
        let (v1mx, v1my) =
            curl_velocity(mx * self.macro_freq, my * self.macro_freq, 2, 1.95, 0.45, s1);
        x += self.macro_advect * v1mx;
        y += self.macro_advect * v1my;

        // Medium wet-on-wet brushstroke edge contour undulation
        let s2 = self.seed ^ 0x3C3C_C3C3_9696_6969;
        let (v2x, v2y) = curl_velocity(x * self.mid_freq, y * self.mid_freq, 2, 2.05, 0.42, s2);
        x += self.mid_advect * v2x;
        y += self.mid_advect * v2y;

        (x, y)
    }

    /// Renders the painterly gradient wallpaper at `(width, height)` in parallel across all CPU cores.
    pub fn render(&self, width: u32, height: u32) -> GeneratedWallpaper {
        let w_usize = width.max(1) as usize;
        let h_usize = height.max(1) as usize;
        let mut rgba = vec![0_u8; w_usize * h_usize * 4];

        let inv_h = 1.0 / height.max(1) as f32;
        let half_w = 0.5 * width as f32;
        let half_h = 0.5 * height as f32;
        let n_colors = self.pigments.len();
        let dither_seed = (self.seed as u32) ^ ((self.seed >> 32) as u32);

        let saduran_seed_a = self.seed ^ 0x1234_5678_9ABC_DEF0;
        let saduran_seed_b = self.seed ^ 0x0FED_CBA9_8765_4321;
        let veil_seed = self.seed ^ 0x7777_3333_BBBB_5555;

        rgba.par_chunks_exact_mut(w_usize * 4)
            .enumerate()
            .for_each(|(py, row)| {
                let y_norm = (py as f32 + 0.5 - half_h) * inv_h;
                let mut weights = [0.0_f32; 7];

                for px in 0..w_usize {
                    let x_norm = (px as f32 + 0.5 - half_w) * inv_h;

                    // 1. Smooth fluid & brushwork domain warp
                    let (wx, wy) = self.warp_point(x_norm, y_norm);

                    // 2. Spatially varying saduran parameters:
                    //    - `delta` controls transition spread (lost sfumato vs defined soft brush edge)
                    //    - `power` controls pigment dominance curve
                    let (mod_a, _, _) = fbm2d_grad(
                        wx * self.saduran_freq,
                        wy * self.saduran_freq,
                        2,
                        2.0,
                        0.5,
                        saduran_seed_a,
                    );
                    let (mod_b, _, _) = fbm2d_grad(
                        wx * (self.saduran_freq * 0.85) + 3.7,
                        wy * (self.saduran_freq * 0.85) - 2.3,
                        2,
                        2.0,
                        0.5,
                        saduran_seed_b,
                    );

                    let delta = (self.delta_base + mod_a * self.delta_var).clamp(0.038, 0.25);
                    let power = (self.power_base + mod_b * self.power_var).clamp(2.15, 4.85);

                    weights[..n_colors].fill(0.0);

                    // 3. Evaluate $C^\infty$-smooth asymmetric distance to each curved brush ribbon
                    for ribbon in &self.ribbons {
                        // Smooth minimum across polyline segments (log-sum-exp with kappa = 14.0)
                        // eliminates any C1 derivative kinks between curve segments!
                        let mut min_d = f32::INFINITY;
                        let mut seg_dists = [0.0_f32; SEGMENTS_PER_RIBBON];

                        for (si, seg) in ribbon.segments.iter().enumerate() {
                            let apx = wx - seg.ax;
                            let apy = wy - seg.ay;
                            let h =
                                ((apx * seg.abx + apy * seg.aby) * seg.inv_len2).clamp(0.0, 1.0);
                            let dx = apx - seg.abx * h;
                            let dy = apy - seg.aby * h;
                            let raw_dist = (dx * dx + dy * dy).sqrt();

                            // Smooth signed side factor in [0, 1] — zero discontinuity across spine!
                            let side = dx * seg.nx + dy * seg.ny;
                            let side_t = 0.5 + 0.5 * (side / (raw_dist + 0.06));
                            let asym_scale =
                                seg.scale_neg + (seg.scale_pos - seg.scale_neg) * side_t;

                            let eff_d = (raw_dist - seg.core_width).max(0.0) * asym_scale;
                            seg_dists[si] = eff_d;
                            if eff_d < min_d {
                                min_d = eff_d;
                            }
                        }

                        const KAPPA: f32 = 14.0;
                        let mut lse_sum = 0.0_f32;
                        for &d in &seg_dists {
                            lse_sum += (-(d - min_d) * KAPPA).exp();
                        }
                        let smooth_dist = (min_d - lse_sum.ln() / KAPPA).max(0.0);

                        // Gaussian-enveloped local inverse-power weight:
                        // Vibrant pure pigment near ribbon core, silky melt between touching neighbors,
                        // and rapid Gaussian roll-off so distant colors never fog the canvas into mud.
                        let reach_env = (-self.reach_decay * smooth_dist * smooth_dist).exp();
                        let contrib =
                            ribbon.presence * reach_env / (smooth_dist + delta).powf(power);
                        weights[ribbon.color_idx] += contrib;
                    }

                    // Normalize weights to partition of unity
                    let mut w_sum = 0.0_f32;
                    for &w in &weights[..n_colors] {
                        w_sum += w;
                    }
                    let inv_w = 1.0 / w_sum.max(1e-8);
                    for w in weights.iter_mut().take(n_colors) {
                        *w *= inv_w;
                    }

                    // 4. Ultra-subtle atmospheric tonal breathing (broad, velvety, never grainy)
                    let (veil, _, _) = simplex2d_grad(wx * 1.35, wy * 1.35, veil_seed);
                    let micro_l = veil * self.atm_veil_amp;
                    let micro_warmth = veil * (self.atm_veil_amp * 0.75);

                    // 5. 16-band Kubelka-Munk + Oklab painterly pigment mix
                    let lin = mix_pigments_painterly(
                        &self.pigments,
                        &weights[..n_colors],
                        self.glaze_bias,
                        micro_l,
                        micro_warmth,
                    );

                    // 6. Blue-noise TPDF dither for zero 8-bit color banding
                    let dither = blue_tpdf_dither(px as u32, py as u32, dither_seed);
                    let srgb = lin.to_srgb8_dithered(dither);

                    let out = &mut row[px * 4..px * 4 + 4];
                    out[0] = srgb.r;
                    out[1] = srgb.g;
                    out[2] = srgb.b;
                    out[3] = 255;
                }
            });

        GeneratedWallpaper {
            width,
            height,
            rgba,
            seed: self.seed,
            style_label: self.style_label,
        }
    }
}

/// Generates `n` maximally separated yet organically asymmetric screen anchors
/// across widescreen space `[-0.74..0.74] x [-0.42..0.42]` via Mitchell's Best-Candidate sampling.
fn generate_organic_anchors(n: usize, rng: &mut StdRng) -> Vec<(f32, f32)> {
    let mut anchors: Vec<(f32, f32)> = Vec::with_capacity(n);

    for idx in 0..n {
        if idx == 0 {
            anchors.push((
                rng.random_range(-0.62..0.62),
                rng.random_range(-0.36..0.36),
            ));
            continue;
        }

        let mut best_candidate = (0.0_f32, 0.0_f32);
        let mut best_score = -1.0_f32;

        for _attempt in 0..36 {
            let cx = rng.random_range(-0.74..0.74);
            let cy = rng.random_range(-0.42..0.42);

            let nearest = anchors
                .iter()
                .map(|&(ax, ay)| ((cx - ax).powi(2) + (cy - ay).powi(2)).sqrt())
                .fold(f32::INFINITY, f32::min);

            // Slight random organic jitter on score so points don't lock into rigid crystals
            let score = nearest * rng.random_range(0.88..1.12);
            if score > best_score {
                best_score = score;
                best_candidate = (cx, cy);
            }
        }
        anchors.push(best_candidate);
    }

    anchors
}

/// Constructs a sweeping Cubic-Bézier brushstroke ribbon passing through `(cx, cy)`
/// with smooth brush-load tapering and continuous asymmetric cross-stroke feathering.
#[allow(clippy::too_many_arguments)]
fn build_curved_ribbon(
    color_idx: usize,
    cx: f32,
    cy: f32,
    angle: f32,
    length: f32,
    core_belly: f32,
    presence: f32,
    rng: &mut StdRng,
) -> BrushRibbon {
    let tx = angle.cos();
    let ty = angle.sin();
    let nx = -ty;
    let ny = tx;

    let half_l = 0.5 * length;
    let bend1 = rng.random_range(-0.36..0.36) * length;
    let bend2 = rng.random_range(-0.36..0.36) * length;
    let end_curl = rng.random_range(-0.24..0.24) * length;

    // Cubic Bézier control points adapted so t ≈ 0.5 passes near `(cx, cy)`
    let p0 = (
        cx - tx * half_l + nx * rng.random_range(-0.12..0.12),
        cy - ty * half_l + ny * rng.random_range(-0.12..0.12),
    );
    let p1 = (
        cx - tx * (half_l * 0.35) + nx * bend1,
        cy - ty * (half_l * 0.35) + ny * bend1,
    );
    let p2 = (
        cx + tx * (half_l * 0.35) + nx * bend2,
        cy + ty * (half_l * 0.35) + ny * bend2,
    );
    let p3 = (
        cx + tx * half_l + nx * end_curl,
        cy + ty * half_l + ny * end_curl,
    );

    // Asymmetric lost-and-found edge scale:
    // Lower distance scale -> wider atmospheric bleed (lost edge)
    // Higher distance scale -> more sculpted soft contour (found edge)
    let flip = rng.random_bool(0.5);
    let found_scale = rng.random_range(1.05..1.55);
    let lost_scale = found_scale * rng.random_range(0.38..0.66);
    let (scale_pos_base, scale_neg_base) = if flip {
        (found_scale, lost_scale)
    } else {
        (lost_scale, found_scale)
    };

    let peak_t = rng.random_range(0.30..0.70);

    let mut pts = [(0.0_f32, 0.0_f32); SEGMENTS_PER_RIBBON + 1];
    for (idx, pt) in pts.iter_mut().enumerate() {
        let t = idx as f32 / SEGMENTS_PER_RIBBON as f32;
        *pt = eval_cubic_bezier(p0, p1, p2, p3, t);
    }

    let dummy = RibbonSegment {
        ax: 0.0,
        ay: 0.0,
        abx: 1.0,
        aby: 0.0,
        inv_len2: 1.0,
        nx: 0.0,
        ny: 1.0,
        core_width: 0.05,
        scale_pos: 1.0,
        scale_neg: 0.6,
    };
    let mut segments = [dummy; SEGMENTS_PER_RIBBON];

    for i in 0..SEGMENTS_PER_RIBBON {
        let a = pts[i];
        let b = pts[i + 1];
        let abx = b.0 - a.0;
        let aby = b.1 - a.1;
        let len2 = (abx * abx + aby * aby).max(1e-6);
        let len = len2.sqrt();

        let t_mid = (i as f32 + 0.5) / SEGMENTS_PER_RIBBON as f32;
        let arch = (PI * t_mid).sin();
        let skew = 1.0 - 0.30 * (t_mid - peak_t).abs();
        let envelope = (arch.powf(0.75) * skew).clamp(0.0, 1.0);

        // Near stroke tips (`arch -> 0`), core width tapers to 0 and distance scale softens
        // so stroke ends feather into mist rather than forming round capsule caps!
        let tip_soften = 0.62 + 0.38 * arch;

        segments[i] = RibbonSegment {
            ax: a.0,
            ay: a.1,
            abx,
            aby,
            inv_len2: 1.0 / len2,
            nx: -aby / len,
            ny: abx / len,
            core_width: core_belly * envelope,
            scale_pos: scale_pos_base * tip_soften,
            scale_neg: scale_neg_base * tip_soften,
        };
    }

    BrushRibbon {
        color_idx,
        presence,
        segments,
    }
}

#[inline]
fn eval_cubic_bezier(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    t: f32,
) -> (f32, f32) {
    let u = 1.0 - t;
    let uu = u * u;
    let uuu = uu * u;
    let tt = t * t;
    let ttt = tt * t;

    let x = uuu * p0.0 + 3.0 * uu * t * p1.0 + 3.0 * u * tt * p2.0 + ttt * p3.0;
    let y = uuu * p0.1 + 3.0 * uu * t * p1.1 + 3.0 * u * tt * p2.1 + ttt * p3.1;
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_distinct_non_uniform_compositions_across_seeds() {
        let palette = [
            Srgb8::new(18, 34, 88),
            Srgb8::new(225, 118, 56),
            Srgb8::new(244, 218, 130),
            Srgb8::new(42, 132, 126),
        ];

        let plan_a = PainterlyPlan::from_palette_with_seed(&palette, 101);
        let plan_b = PainterlyPlan::from_palette_with_seed(&palette, 999_983);

        let img_a = plan_a.render(160, 90);
        let img_b = plan_b.render(160, 90);

        assert_eq!(img_a.rgba.len(), 160 * 90 * 4);
        assert_ne!(img_a.rgba, img_b.rgba);

        let mut diff_acc = 0_u64;
        for i in (0..img_a.rgba.len()).step_by(4) {
            diff_acc += (img_a.rgba[i] as i32 - img_b.rgba[i] as i32).unsigned_abs() as u64;
        }
        assert!(diff_acc > 50_000, "Seeds should produce genuinely distinct compositions");
    }
}
