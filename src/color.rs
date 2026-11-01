//! Painterly color science: Oklab perceptual space + 16-band Kubelka-Munk
//! spectral pigment mixing with exact Moore-Penrose RGB round-trip projection.

use std::sync::OnceLock;

/// 8-bit sRGB color used by UI pickers and final pixel buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Srgb8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Srgb8 {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn to_hex(self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    pub fn from_hex(input: &str) -> Option<Self> {
        let trimmed = input.trim().trim_start_matches('#');
        if trimmed.len() == 6 {
            let r = u8::from_str_radix(&trimmed[0..2], 16).ok()?;
            let g = u8::from_str_radix(&trimmed[2..4], 16).ok()?;
            let b = u8::from_str_radix(&trimmed[4..6], 16).ok()?;
            Some(Self { r, g, b })
        } else if trimmed.len() == 3 {
            let r = u8::from_str_radix(&trimmed[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&trimmed[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&trimmed[2..3], 16).ok()? * 17;
            Some(Self { r, g, b })
        } else {
            None
        }
    }

    /// Convert to HSV where `h` is in `[0.0, 360.0)`, `s` and `v` in `[0.0, 1.0]`.
    pub fn to_hsv(self) -> (f32, f32, f32) {
        let r = self.r as f32 / 255.0;
        let g = self.g as f32 / 255.0;
        let b = self.b as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        let h = if delta <= 1e-6 {
            0.0
        } else if (max - r).abs() <= 1e-6 {
            60.0 * (((g - b) / delta).rem_euclid(6.0))
        } else if (max - g).abs() <= 1e-6 {
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            60.0 * (((r - g) / delta) + 4.0)
        };

        let s = if max <= 1e-6 { 0.0 } else { delta / max };
        (h, s, max)
    }

    /// Construct from HSV (`h` in degrees, `s` and `v` in `[0.0, 1.0]`).
    pub fn from_hsv(h: f32, s: f32, v: f32) -> Self {
        let h = h.rem_euclid(360.0);
        let s = s.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        let c = v * s;
        let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
        let m = v - c;

        let (r1, g1, b1) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        Self {
            r: ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
            g: ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
            b: ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        }
    }

    pub fn relative_luminance(self) -> f32 {
        let lin = LinRgb::from_srgb8(self);
        0.2126 * lin.r + 0.7152 * lin.g + 0.0722 * lin.b
    }
}

/// Linear light RGB in `[0.0, 1.0]`.
#[derive(Clone, Copy, Debug)]
pub struct LinRgb {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[inline]
fn srgb_to_linear_channel(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb_channel(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

impl LinRgb {
    pub fn from_srgb8(c: Srgb8) -> Self {
        Self {
            r: srgb_to_linear_channel(c.r as f32 / 255.0),
            g: srgb_to_linear_channel(c.g as f32 / 255.0),
            b: srgb_to_linear_channel(c.b as f32 / 255.0),
        }
    }

    /// Convert to 8-bit sRGB with triangular-PDF dither (`dither` in `[-0.5, 0.5]` + `[-0.5, 0.5]`)
    /// to eliminate any 8-bit gradient banding.
    #[inline]
    pub fn to_srgb8_dithered(self, tpdf_dither: f32) -> Srgb8 {
        let sr = linear_to_srgb_channel(self.r) * 255.0 + tpdf_dither;
        let sg = linear_to_srgb_channel(self.g) * 255.0 + tpdf_dither;
        let sb = linear_to_srgb_channel(self.b) * 255.0 + tpdf_dither;
        Srgb8 {
            r: sr.round().clamp(0.0, 255.0) as u8,
            g: sg.round().clamp(0.0, 255.0) as u8,
            b: sb.round().clamp(0.0, 255.0) as u8,
        }
    }
}

/// Perceptual Oklab color space (Björn Ottosson).
#[derive(Clone, Copy, Debug)]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Oklab {
    #[inline]
    pub fn from_lin_rgb(c: LinRgb) -> Self {
        // Clamp slightly above zero for out-of-gamut spectral intermediates
        let r = c.r.max(0.0);
        let g = c.g.max(0.0);
        let b = c.b.max(0.0);

        let l = 0.41222147 * r + 0.53633254 * g + 0.051445993 * b;
        let m = 0.2119035 * r + 0.6806995 * g + 0.10739696 * b;
        let s = 0.08830246 * r + 0.28171884 * g + 0.6299787 * b;

        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();

        Self {
            l: 0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_,
            a: 1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_,
            b: 0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_,
        }
    }

    #[inline]
    pub fn to_lin_rgb_raw(self) -> LinRgb {
        let l_ = self.l + 0.39633778 * self.a + 0.21580376 * self.b;
        let m_ = self.l - 0.105561346 * self.a - 0.06385417 * self.b;
        let s_ = self.l - 0.08948418 * self.a - 1.2914855 * self.b;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        LinRgb {
            r: 4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
            g: -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
            b: -0.0041960863 * l - 0.7034186 * m + 1.7076147 * s,
        }
    }

    /// Smoothly compresses Oklab chroma into the sRGB gamut while preserving hue and lightness.
    #[inline]
    pub fn to_gamut_mapped_lin_rgb(mut self) -> LinRgb {
        self.l = self.l.clamp(0.0, 1.0);
        let raw = self.to_lin_rgb_raw();
        const EPS: f32 = 1.5e-3;
        if raw.r >= -EPS
            && raw.r <= 1.0 + EPS
            && raw.g >= -EPS
            && raw.g <= 1.0 + EPS
            && raw.b >= -EPS
            && raw.b <= 1.0 + EPS
        {
            return LinRgb {
                r: raw.r.clamp(0.0, 1.0),
                g: raw.g.clamp(0.0, 1.0),
                b: raw.b.clamp(0.0, 1.0),
            };
        }

        // Fast 6-step binary search along chroma ray to preserve exact perceptual hue
        let mut lo = 0.0_f32;
        let mut hi = 1.0_f32;
        for _ in 0..6 {
            let mid = 0.5 * (lo + hi);
            let test = Oklab {
                l: self.l,
                a: self.a * mid,
                b: self.b * mid,
            }
            .to_lin_rgb_raw();
            if test.r >= -EPS
                && test.r <= 1.0 + EPS
                && test.g >= -EPS
                && test.g <= 1.0 + EPS
                && test.b >= -EPS
                && test.b <= 1.0 + EPS
            {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let mapped = Oklab {
            l: self.l,
            a: self.a * lo,
            b: self.b * lo,
        }
        .to_lin_rgb_raw();

        LinRgb {
            r: mapped.r.clamp(0.0, 1.0),
            g: mapped.g.clamp(0.0, 1.0),
            b: mapped.b.clamp(0.0, 1.0),
        }
    }
}

/// Number of spectral wavelength bands from 400nm to 700nm.
const BANDS: usize = 16;

/// Small reflectance floor representing pigment scattering limit (prevents infinite K/S at black
/// while keeping tinting strength realistic like oil/acrylic pigments).
const R_FLOOR: f32 = 0.012;
const R_CEIL: f32 = 0.985;

struct SpectralTables {
    /// Basis vectors `B[k] = [Br(lambda_k), Bg(lambda_k), Bb(lambda_k)]`, summing to 1.0 per band.
    basis: [[f32; 3]; BANDS],
    /// Exact Moore-Penrose pseudoinverse `P` (`3 x BANDS`) such that `P * B = I_3`.
    proj: [[f32; BANDS]; 3],
}

static SPECTRAL_TABLES: OnceLock<SpectralTables> = OnceLock::new();

fn spectral_tables() -> &'static SpectralTables {
    SPECTRAL_TABLES.get_or_init(|| {
        let mut basis = [[0.0_f32; 3]; BANDS];

        for (k, row) in basis.iter_mut().enumerate() {
            let lambda = 400.0 + (k as f64) * (300.0 / (BANDS - 1) as f64);

            // Smooth overlapping pigment reflectance lobes:
            // - Blue peaks at 442nm and tails into cyan-green (~535nm)
            // - Green peaks at 534nm, overlapping both blue-cyan and yellow-red
            // - Red peaks at 636nm and rises from yellow-orange (~555nm)
            let wb = (-0.5 * ((lambda - 442.0) / 44.0).powi(2)).exp()
                + 0.18 * (-0.5 * ((lambda - 492.0) / 28.0).powi(2)).exp();
            let wg = (-0.5 * ((lambda - 534.0) / 42.0).powi(2)).exp();
            let wr = (-0.5 * ((lambda - 636.0) / 48.0).powi(2)).exp()
                + 0.20 * (-0.5 * ((lambda - 584.0) / 28.0).powi(2)).exp();

            let sum = wb + wg + wr;
            row[0] = (wr / sum) as f32;
            row[1] = (wg / sum) as f32;
            row[2] = (wb / sum) as f32;
        }

        // Compute Gram matrix G = B^T * B (3x3) in f64 for exact inversion
        let mut g = [[0.0_f64; 3]; 3];
        for row in &basis {
            for i in 0..3 {
                for j in 0..3 {
                    g[i][j] += (row[i] as f64) * (row[j] as f64);
                }
            }
        }

        let g_inv = invert_3x3(g);

        // Pseudoinverse P = G^-1 * B^T (3 x BANDS)
        let mut proj = [[0.0_f32; BANDS]; 3];
        for i in 0..3 {
            for k in 0..BANDS {
                let mut acc = 0.0_f64;
                for j in 0..3 {
                    acc += g_inv[i][j] * (basis[k][j] as f64);
                }
                proj[i][k] = acc as f32;
            }
        }

        SpectralTables { basis, proj }
    })
}

fn invert_3x3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let inv_det = 1.0 / det;

    [
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ]
}

/// Precomputed pigment entry for fast per-pixel multi-color blending.
#[derive(Clone, Copy, Debug)]
pub struct PreparedPigment {
    pub oklab: Oklab,
    /// Kubelka-Munk absorption/scattering ratio `K/S` across 16 spectral bands.
    pub ks: [f32; BANDS],
    /// Chroma magnitude in Oklab (`sqrt(a^2 + b^2)`).
    pub chroma: f32,
}

impl PreparedPigment {
    pub fn from_srgb8(color: Srgb8) -> Self {
        let tables = spectral_tables();
        let lin = LinRgb::from_srgb8(color);
        let oklab = Oklab::from_lin_rgb(lin);
        let chroma = (oklab.a * oklab.a + oklab.b * oklab.b).sqrt();

        let mut ks = [0.0_f32; BANDS];
        for (ks_k, b) in ks.iter_mut().zip(tables.basis.iter()) {
            let r_raw = lin.r * b[0] + lin.g * b[1] + lin.b * b[2];
            let r_clamped = R_FLOOR + r_raw.clamp(0.0, 1.0) * (R_CEIL - R_FLOOR);
            let one_minus_r = 1.0 - r_clamped;
            *ks_k = (one_minus_r * one_minus_r) / (2.0 * r_clamped);
        }

        Self { oklab, ks, chroma }
    }
}

/// Mix up to 7 prepared pigments with normalized weights `weights` (`sum == 1.0`).
///
/// Combines Kubelka-Munk subtractive pigment physics (for authentic chromatic undertones,
/// e.g. Blue + Yellow -> Emerald/Sage, Crimson + Ultramarine -> Deep Violet) with Oklab
/// perceptual glazing so transitions retain painterly luminosity and vibrancy.
#[inline]
pub fn mix_pigments_painterly(
    pigments: &[PreparedPigment],
    weights: &[f32],
    glaze_bias: f32,
    micro_l: f32,
    micro_warmth: f32,
) -> LinRgb {
    debug_assert_eq!(pigments.len(), weights.len());
    let tables = spectral_tables();

    // 1. Weighted Kubelka-Munk K/S spectrum & weighted Oklab perceptual target
    let mut ks_mix = [0.0_f32; BANDS];
    let mut ok_l = 0.0_f32;
    let mut ok_a = 0.0_f32;
    let mut ok_b = 0.0_f32;
    let mut avg_chroma = 0.0_f32;
    let mut max_weight = 0.0_f32;
    let mut weight_sq_sum = 0.0_f32;

    for (p, &w) in pigments.iter().zip(weights.iter()) {
        if w <= 1e-6 {
            continue;
        }
        max_weight = max_weight.max(w);
        weight_sq_sum += w * w;
        ok_l += w * p.oklab.l;
        ok_a += w * p.oklab.a;
        ok_b += w * p.oklab.b;
        avg_chroma += w * p.chroma;
        for (km, &pk) in ks_mix.iter_mut().zip(p.ks.iter()) {
            *km += w * pk;
        }
    }

    // 2. Invert Kubelka-Munk back to spectral reflectance R_mix(lambda)
    let mut r_spec = [0.0_f32; BANDS];
    let inv_span = 1.0 / (R_CEIL - R_FLOOR);
    for (rs, &ks_val) in r_spec.iter_mut().zip(ks_mix.iter()) {
        let ks = ks_val.max(0.0);
        let r_clamped = 1.0 + ks - (ks * ks + 2.0 * ks).sqrt();
        *rs = ((r_clamped - R_FLOOR) * inv_span).clamp(0.0, 1.0);
    }

    // 3. Project reflectance back to linear RGB via exact Moore-Penrose pseudoinverse
    let mut km_r = 0.0_f32;
    let mut km_g = 0.0_f32;
    let mut km_b = 0.0_f32;
    for (k, &rk) in r_spec.iter().enumerate() {
        km_r += tables.proj[0][k] * rk;
        km_g += tables.proj[1][k] * rk;
        km_b += tables.proj[2][k] * rk;
    }

    // Convert KM result to Oklab; preserve signed chromaticity gently even if slightly out-of-gamut
    let km_oklab = Oklab::from_lin_rgb(LinRgb {
        r: km_r.max(0.0),
        g: km_g.max(0.0),
        b: km_b.max(0.0),
    });

    // 4. Painterly Glaze Harmonization:
    // Effective pigment count N_eff = 1 / sum(w_i^2):
    // - N_eff ≈ 1.0: pure pigment
    // - N_eff ≈ 2.0: clean 2-pigment blend -> strong Kubelka-Munk pigment physics (e.g. Blue+Yellow -> Emerald)
    // - N_eff >= 2.6: 3+ pigment junction -> shift smoothly toward luminous Oklab glazing to avoid subtractive mud!
    let n_eff = 1.0 / weight_sq_sum.max(0.14);
    let multi_junction = ((n_eff - 2.15) * 0.75).clamp(0.0, 1.0);

    let mix_zone = (1.0 - max_weight).clamp(0.0, 1.0);
    let base_km_share = (0.72 + 0.12 * glaze_bias).clamp(0.50, 0.86);
    let km_share = base_km_share * (1.0 - 0.55 * multi_junction);

    // Lightness: preserve vibrant luminosity especially at multi-color junctions
    let km_l_weight = (0.52 + 0.10 * glaze_bias) * (1.0 - 0.65 * multi_junction);
    let l_blend = km_oklab.l * km_l_weight + ok_l * (1.0 - km_l_weight);
    let mut final_l = ok_l + (l_blend - ok_l) * (mix_zone * 1.15).clamp(0.0, 1.0);

    let mut final_a = km_oklab.a * km_share + ok_a * (1.0 - km_share);
    let mut final_b = km_oklab.b * km_share + ok_b * (1.0 - km_share);

    // Chroma vitality rescue: keep painterly transitions rich & jewel-toned rather than dull gray
    let cur_chroma = (final_a * final_a + final_b * final_b).sqrt();
    let min_target_chroma = avg_chroma * (0.56 + 0.12 * multi_junction) * mix_zone;
    if cur_chroma > 1e-4 && cur_chroma < min_target_chroma {
        let boost = (min_target_chroma / cur_chroma).min(2.15);
        final_a *= boost;
        final_b *= boost;
    }

    // Apply subtle organic brushwork tonal & warmth micro-nuance
    final_l = (final_l + micro_l).clamp(0.0, 1.0);
    final_a += micro_warmth * 0.008;
    final_b += micro_warmth * 0.011;

    Oklab {
        l: final_l,
        a: final_a,
        b: final_b,
    }
    .to_gamut_mapped_lin_rgb()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_single_pigment_roundtrip() {
        let samples = [
            Srgb8::new(255, 0, 0),
            Srgb8::new(0, 255, 0),
            Srgb8::new(0, 0, 255),
            Srgb8::new(245, 210, 45),
            Srgb8::new(24, 58, 140),
            Srgb8::new(255, 255, 255),
            Srgb8::new(15, 15, 18),
        ];

        for s in samples {
            let p = PreparedPigment::from_srgb8(s);
            let lin = mix_pigments_painterly(&[p], &[1.0], 0.0, 0.0, 0.0);
            let out = lin.to_srgb8_dithered(0.0);
            assert!(
                (out.r as i32 - s.r as i32).abs() <= 1
                    && (out.g as i32 - s.g as i32).abs() <= 1
                    && (out.b as i32 - s.b as i32).abs() <= 1,
                "Expected {:?}, got {:?}",
                s,
                out
            );
        }
    }

    #[test]
    fn blue_and_yellow_mix_into_painterly_green_not_gray() {
        let blue = PreparedPigment::from_srgb8(Srgb8::new(15, 65, 215));
        let yellow = PreparedPigment::from_srgb8(Srgb8::new(245, 215, 20));
        let mixed = mix_pigments_painterly(&[blue, yellow], &[0.5, 0.5], 0.0, 0.0, 0.0)
            .to_srgb8_dithered(0.0);

        // In painterly pigment mixing, Green must clearly dominate both Red and Blue!
        assert!(
            mixed.g > mixed.r + 18 && mixed.g > mixed.b + 18,
            "Expected lush green undertone from blue + yellow, got {:?}",
            mixed
        );
    }
}

