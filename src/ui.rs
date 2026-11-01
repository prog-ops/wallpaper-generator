//! Bespoke "Atelier Chroma" GUI:
//! - Grand opening palette studio for choosing 2..=7 colors with inline GPU-rendered
//!   Saturation-Value & Hue color pickers per slot.
//! - Full-bleed painterly wallpaper canvas view with one-click random regeneration,
//!   unobstructed preview toggle, and high-res PNG export.

use std::fs::File;
use std::io::BufWriter;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, CornerRadius, CursorIcon, FontFamily, FontId, Id,
    Key, Layout, Margin, Mesh, Order, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Ui, Vec2,
};
use rand::prelude::*;

use crate::color::Srgb8;
use crate::painter::{GeneratedWallpaper, PainterlyPlan};

const MIN_COLORS: usize = 2;
const MAX_COLORS: usize = 7;

/// Default crisp wallpaper render resolution (QHD 16:9 — sharp on both 1080p and 1440p displays).
const CANVAS_W: u32 = 2560;
const CANVAS_H: u32 = 1440;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppScreen {
    PaletteMenu,
    WallpaperCanvas,
}

/// Floating-point HSV state per picker slot so Hue never jumps when Saturation or Value reaches 0.
#[derive(Clone)]
struct ColorSlot {
    h: f32,
    s: f32,
    v: f32,
    hex_input: String,
    hex_editing: bool,
}

impl ColorSlot {
    fn from_srgb(c: Srgb8) -> Self {
        let (h, s, v) = c.to_hsv();
        Self {
            h,
            s,
            v,
            hex_input: c.to_hex(),
            hex_editing: false,
        }
    }

    fn srgb(&self) -> Srgb8 {
        Srgb8::from_hsv(self.h, self.s, self.v)
    }

    fn sync_hex(&mut self) {
        if !self.hex_editing {
            self.hex_input = self.srgb().to_hex();
        }
    }

    fn egui_color(&self) -> Color32 {
        let c = self.srgb();
        Color32::from_rgb(c.r, c.g, c.b)
    }
}

enum RenderJobResult {
    Preview(GeneratedWallpaper),
    ExportSaved(Result<String, String>),
}

pub struct AtelierApp {
    screen: AppScreen,
    color_count: usize,
    slots: [ColorSlot; MAX_COLORS],

    // Background rendering channel & live wallpaper texture
    tx: Sender<RenderJobResult>,
    rx: Receiver<RenderJobResult>,
    is_generating: bool,
    is_exporting: bool,

    current_wallpaper: Option<GeneratedWallpaper>,
    wallpaper_texture: Option<TextureHandle>,
    hide_canvas_hud: bool,
    status_toast: Option<(String, f64)>,
}

impl AtelierApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_studio_visuals(&cc.egui_ctx);

        let (tx, rx) = mpsc::channel();

        // Curated painterly starting palette across 7 slots (default active count = 4)
        let initial_palette = [
            Srgb8::new(22, 43, 92),   // Prussian Nocturne
            Srgb8::new(219, 109, 64), // Venetian Terracotta
            Srgb8::new(242, 206, 122), // Naples Ochre
            Srgb8::new(45, 134, 128), // Viridian Sea
            Srgb8::new(146, 58, 94),  // Alizarin Plum
            Srgb8::new(236, 229, 216), // Warm Titanium Parchment
            Srgb8::new(34, 28, 52),   // Umber Indigo
        ];

        Self {
            screen: AppScreen::PaletteMenu,
            color_count: 4,
            slots: initial_palette.map(ColorSlot::from_srgb),
            tx,
            rx,
            is_generating: false,
            is_exporting: false,
            current_wallpaper: None,
            wallpaper_texture: None,
            hide_canvas_hud: false,
            status_toast: None,
        }
    }

    fn active_palette(&self) -> Vec<Srgb8> {
        self.slots[..self.color_count]
            .iter()
            .map(ColorSlot::srgb)
            .collect()
    }

    fn trigger_generate(&mut self, ctx: &egui::Context) {
        if self.is_generating {
            return;
        }
        self.is_generating = true;
        self.hide_canvas_hud = false;

        let palette = self.active_palette();
        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();

        thread::spawn(move || {
            let plan = PainterlyPlan::random_from_palette(&palette);
            let rendered = plan.render(CANVAS_W, CANVAS_H);
            let _ = tx.send(RenderJobResult::Preview(rendered));
            ctx_clone.request_repaint();
        });
    }

    fn trigger_export_png(&mut self, ctx: &egui::Context) {
        let Some(wp) = self.current_wallpaper.clone() else {
            return;
        };
        if self.is_exporting {
            return;
        }
        self.is_exporting = true;

        let tx = self.tx.clone();
        let ctx_clone = ctx.clone();

        thread::spawn(move || {
            let default_name = format!("painterly_wallpaper_{:08X}.png", wp.seed as u32);
            let picked = rfd::FileDialog::new()
                .set_title("Simpan Wallpaper PNG")
                .set_file_name(&default_name)
                .add_filter("PNG Image", &["png"])
                .save_file();

            if let Some(path) = picked {
                let res = save_rgba_png(&path, wp.width, wp.height, &wp.rgba)
                    .map(|_| path.display().to_string())
                    .map_err(|e| format!("Gagal menyimpan PNG: {e}"));
                let _ = tx.send(RenderJobResult::ExportSaved(res));
            } else {
                let _ = tx.send(RenderJobResult::ExportSaved(Err(String::new())));
            }
            ctx_clone.request_repaint();
        });
    }

    fn randomize_harmonious_palette(&mut self) {
        let mut rng = rand::rng();
        let base_hue = rng.random_range(0.0..360.0_f32);
        let scheme = rng.random_range(0..4);

        for i in 0..self.color_count {
            let t = i as f32 / (self.color_count.max(2) - 1) as f32;
            let hue_offset = match scheme {
                // Split-complementary painterly harmony
                0 => [0.0, 28.0, 155.0, 195.0, 335.0, 52.0, 220.0][i],
                // Analogous atmospheric sweep + warm accent
                1 => t * 115.0 + if i == self.color_count - 1 { 165.0 } else { 0.0 },
                // Triadic museum palette
                2 => (i as f32 * 118.0) + rng.random_range(-16.0..16.0),
                // Golden-angle organic dispersion
                _ => i as f32 * 137.508,
            };

            let h = (base_hue + hue_offset + rng.random_range(-10.0..10.0)).rem_euclid(360.0);
            // Vary value & saturation like a real painter (deep anchor, midtones, luminous highlight)
            let (s, v) = match i % 4 {
                0 => (rng.random_range(0.58..0.86), rng.random_range(0.22..0.44)),
                1 => (rng.random_range(0.52..0.84), rng.random_range(0.66..0.92)),
                2 => (rng.random_range(0.25..0.58), rng.random_range(0.84..0.98)),
                _ => (rng.random_range(0.45..0.78), rng.random_range(0.45..0.76)),
            };

            self.slots[i].h = h;
            self.slots[i].s = s;
            self.slots[i].v = v;
            self.slots[i].hex_editing = false;
            self.slots[i].sync_hex();
        }
    }

    fn poll_background_jobs(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                RenderJobResult::Preview(wp) => {
                    let color_image = ColorImage::from_rgba_unmultiplied(
                        [wp.width as usize, wp.height as usize],
                        &wp.rgba,
                    );
                    if let Some(tex) = &mut self.wallpaper_texture {
                        tex.set(color_image, TextureOptions::LINEAR);
                    } else {
                        self.wallpaper_texture = Some(ctx.load_texture(
                            "painterly_wallpaper",
                            color_image,
                            TextureOptions::LINEAR,
                        ));
                    }
                    self.current_wallpaper = Some(wp);
                    self.is_generating = false;
                    self.screen = AppScreen::WallpaperCanvas;
                }
                RenderJobResult::ExportSaved(res) => {
                    self.is_exporting = false;
                    let now = ctx.input(|i| i.time);
                    match res {
                        Ok(path) => {
                            self.status_toast = Some((format!("Tersimpan: {path}"), now + 4.0));
                        }
                        Err(err) if !err.is_empty() => {
                            self.status_toast = Some((err, now + 4.5));
                        }
                        Err(_) => {}
                    }
                }
            }
        }
    }
}

impl eframe::App for AtelierApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_background_jobs(ctx);

        // Keyboard shortcuts on Wallpaper view
        if self.screen == AppScreen::WallpaperCanvas {
            let (pressed_esc, pressed_h, pressed_space) = ctx.input(|i| {
                (
                    i.key_pressed(Key::Escape),
                    i.key_pressed(Key::H),
                    i.key_pressed(Key::Space),
                )
            });
            if pressed_esc {
                if self.hide_canvas_hud {
                    self.hide_canvas_hud = false;
                } else {
                    self.screen = AppScreen::PaletteMenu;
                }
            } else if pressed_h {
                self.hide_canvas_hud = !self.hide_canvas_hud;
            } else if pressed_space && !self.is_generating {
                self.trigger_generate(ctx);
            }
        }

        match self.screen {
            AppScreen::PaletteMenu => self.draw_grand_palette_menu(ctx),
            AppScreen::WallpaperCanvas => self.draw_wallpaper_showcase(ctx),
        }
    }
}

impl AtelierApp {
    /// Grand opening Studio Menu where users choose `2..=7` colors and tune each inline color picker.
    fn draw_grand_palette_menu(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::from_rgb(12, 13, 18)))
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                self.paint_ambient_studio_backdrop(ui, rect);

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let avail_w = ui.available_width();
                        let content_w = avail_w.min(1340.0);
                        let side_pad = ((avail_w - content_w) * 0.5).max(20.0);

                        ui.add_space(28.0);
                        ui.horizontal(|ui| {
                            ui.add_space(side_pad);
                            ui.vertical(|ui| {
                                ui.set_width(content_w - side_pad.min(20.0));
                                self.draw_menu_header(ui);
                                ui.add_space(20.0);
                                self.draw_color_count_bar(ui);
                                ui.add_space(22.0);
                                self.draw_color_picker_gallery(ui);
                                ui.add_space(26.0);
                                self.draw_generate_dock(ctx, ui);
                                ui.add_space(32.0);
                            });
                        });
                    });
            });
    }

    /// Subtle dark gallery backdrop softly illuminated by the user's chosen palette colors.
    fn paint_ambient_studio_backdrop(&self, ui: &Ui, rect: Rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, 0.0, Color32::from_rgb(11, 12, 17));

        // Build a soft 4x3 vertex mesh glow tinted by the active palette
        let cols = 4_usize;
        let rows = 3_usize;
        let mut mesh = Mesh::default();

        for r in 0..rows {
            let fy = r as f32 / (rows - 1) as f32;
            for c in 0..cols {
                let fx = c as f32 / (cols - 1) as f32;
                let pos = Pos2::new(
                    rect.left() + fx * rect.width(),
                    rect.top() + fy * rect.height(),
                );

                let slot_idx = (c + r * 2) % self.color_count;
                let srgb = self.slots[slot_idx].srgb();
                // Keep backdrop deep velvet (12-16% mix over dark obsidian) so pickers stay accurate
                let glow = if (r == 0 && c == 1) || (r == 2 && c == 2) || (r == 1 && c == 3) {
                    0.15
                } else {
                    0.06
                };
                let cr = (11.0 + (srgb.r as f32 - 11.0) * glow) as u8;
                let cg = (12.0 + (srgb.g as f32 - 12.0) * glow) as u8;
                let cb = (17.0 + (srgb.b as f32 - 17.0) * glow) as u8;

                mesh.vertices.push(egui::epaint::Vertex {
                    pos,
                    uv: Pos2::ZERO,
                    color: Color32::from_rgb(cr, cg, cb),
                });
            }
        }

        for r in 0..(rows - 1) {
            for c in 0..(cols - 1) {
                let i0 = (r * cols + c) as u32;
                let i1 = i0 + 1;
                let i2 = i0 + cols as u32;
                let i3 = i2 + 1;
                mesh.add_triangle(i0, i1, i2);
                mesh.add_triangle(i1, i3, i2);
            }
        }
        painter.add(mesh);
    }

    fn draw_menu_header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new("ATELIER CHROMA")
                        .font(FontId::new(12.5, FontFamily::Proportional))
                        .color(Color32::from_rgb(168, 174, 198))
                        .strong(),
                );
                ui.add_space(3.0);
                ui.label(
                    RichText::new("Painterly Gradient Studio")
                        .font(FontId::new(31.0, FontFamily::Proportional))
                        .color(Color32::from_rgb(245, 246, 250))
                        .strong(),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(
                        "Racik 2 hingga 7 pigmen warna. Setiap klik Generate! melukis saduran organik & komposisi aliran baru yang tak pernah berulang.",
                    )
                    .font(FontId::new(14.0, FontFamily::Proportional))
                    .color(Color32::from_rgb(154, 160, 180)),
                );
            });

            if self.current_wallpaper.is_some() {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let back_btn = egui::Button::new(
                        RichText::new("Lihat Lukisan Terakhir  →")
                            .font(FontId::new(13.5, FontFamily::Proportional))
                            .color(Color32::from_rgb(232, 235, 245)),
                    )
                    .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 16))
                    .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 36)))
                    .corner_radius(CornerRadius::same(10))
                    .min_size(Vec2::new(178.0, 38.0));

                    if ui.add(back_btn).on_hover_cursor(CursorIcon::PointingHand).clicked() {
                        self.screen = AppScreen::WallpaperCanvas;
                    }
                });
            }
        });
    }

    /// Prominent selector for choosing `2`, `3`, `4`, `5`, `6`, or `7` palette colors.
    fn draw_color_count_bar(&mut self, ui: &mut Ui) {
        egui::Frame::NONE
            .fill(Color32::from_rgba_unmultiplied(22, 24, 34, 215))
            .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 22)))
            .corner_radius(CornerRadius::same(16))
            .inner_margin(Margin::symmetric(20, 14))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new("Jumlah Kombinasi Warna")
                            .font(FontId::new(15.0, FontFamily::Proportional))
                            .color(Color32::from_rgb(225, 229, 242))
                            .strong(),
                    );
                    ui.add_space(14.0);

                    for count in MIN_COLORS..=MAX_COLORS {
                        let selected = self.color_count == count;
                        let fill = if selected {
                            Color32::from_rgb(240, 242, 250)
                        } else {
                            Color32::from_rgba_unmultiplied(255, 255, 255, 12)
                        };
                        let text_col = if selected {
                            Color32::from_rgb(14, 16, 24)
                        } else {
                            Color32::from_rgb(198, 204, 222)
                        };
                        let border = if selected {
                            Stroke::new(1.5, Color32::WHITE)
                        } else {
                            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 24))
                        };

                        let btn = egui::Button::new(
                            RichText::new(format!("{count} Warna"))
                                .font(FontId::new(14.0, FontFamily::Proportional))
                                .color(text_col)
                                .strong(),
                        )
                        .fill(fill)
                        .stroke(border)
                        .corner_radius(CornerRadius::same(10))
                        .min_size(Vec2::new(86.0, 36.0));

                        if ui
                            .add(btn)
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .clicked()
                        {
                            self.color_count = count;
                        }
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let shuffle_btn = egui::Button::new(
                            RichText::new("🎲 Inspirasi Palet")
                                .font(FontId::new(13.5, FontFamily::Proportional))
                                .color(Color32::from_rgb(224, 228, 242)),
                        )
                        .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 14))
                        .stroke(Stroke::new(
                            1.0,
                            Color32::from_rgba_unmultiplied(255, 255, 255, 28),
                        ))
                        .corner_radius(CornerRadius::same(10))
                        .min_size(Vec2::new(142.0, 36.0));

                        if ui
                            .add(shuffle_btn)
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("Acak kombinasi warna harmonis sebagai titik awal")
                            .clicked()
                        {
                            self.randomize_harmonious_palette();
                        }
                    });
                });
            });
    }

    /// Responsive gallery of inline Color Picker cards for every active slot (`2..=7`).
    fn draw_color_picker_gallery(&mut self, ui: &mut Ui) {
        let total = self.color_count;
        let avail_w = ui.available_width();
        let gap = 16.0_f32;

        // Smart balanced rows so cards are always spacious and centered:
        // 2..=4 colors -> single row (if width >= 920) else 2 rows
        // 5 colors -> row of 3 + centered row of 2
        // 6 colors -> 2 rows of 3
        // 7 colors -> row of 4 + centered row of 3
        let rows: Vec<(usize, usize)> = if avail_w < 860.0 {
            (0..total)
                .step_by(2)
                .map(|start| (start, (start + 2).min(total)))
                .collect()
        } else {
            match total {
                2..=4 => vec![(0, total)],
                5 => vec![(0, 3), (3, 5)],
                6 => vec![(0, 3), (3, 6)],
                _ => vec![(0, 4), (4, 7)],
            }
        };

        let max_cols_in_any_row = rows.iter().map(|&(s, e)| e - s).max().unwrap_or(1);
        let card_w = ((avail_w - gap * (max_cols_in_any_row as f32 - 1.0))
            / max_cols_in_any_row as f32)
            .clamp(220.0, 360.0);

        for (row_idx, (start, end)) in rows.into_iter().enumerate() {
            if row_idx > 0 {
                ui.add_space(gap);
            }
            let count_in_row = end - start;
            let row_total_w =
                count_in_row as f32 * card_w + (count_in_row.saturating_sub(1) as f32) * gap;
            let left_offset = ((avail_w - row_total_w) * 0.5).max(0.0);

            ui.horizontal(|ui| {
                if left_offset > 0.5 {
                    ui.add_space(left_offset);
                }
                for (pos_in_row, slot_idx) in (start..end).enumerate() {
                    if pos_in_row > 0 {
                        ui.add_space(gap);
                    }
                    draw_single_color_picker_card(ui, slot_idx, &mut self.slots[slot_idx], card_w);
                }
            });
        }
    }

    /// Palette summary strip + Hero `Generate!` button at the bottom of the Grand Menu.
    fn draw_generate_dock(&mut self, ctx: &egui::Context, ui: &mut Ui) {
        egui::Frame::NONE
            .fill(Color32::from_rgba_unmultiplied(22, 24, 34, 230))
            .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 28)))
            .corner_radius(CornerRadius::same(18))
            .inner_margin(Margin::symmetric(22, 18))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Live Pigment Harmony Strip on the left
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("Palet Pigmen Aktif")
                                .font(FontId::new(12.5, FontFamily::Proportional))
                                .color(Color32::from_rgb(158, 164, 184)),
                        );
                        ui.add_space(6.0);
                        let strip_w = (ui.available_width() - 260.0).clamp(180.0, 620.0);
                        let (strip_rect, _) =
                            ui.allocate_exact_size(Vec2::new(strip_w, 30.0), Sense::hover());
                        paint_pigment_preview_strip(ui, strip_rect, &self.slots[..self.color_count]);
                    });

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let label_text = if self.is_generating {
                            "Melukis Saduran..."
                        } else {
                            "✦  Generate!"
                        };

                        let gen_btn = egui::Button::new(
                            RichText::new(label_text)
                                .font(FontId::new(18.5, FontFamily::Proportional))
                                .color(Color32::from_rgb(12, 14, 20))
                                .strong(),
                        )
                        .fill(Color32::from_rgb(245, 246, 252))
                        .stroke(Stroke::new(1.5, Color32::WHITE))
                        .corner_radius(CornerRadius::same(14))
                        .min_size(Vec2::new(224.0, 54.0));

                        let resp = ui
                            .add_enabled(!self.is_generating, gen_btn)
                            .on_hover_cursor(CursorIcon::PointingHand);
                        if resp.clicked() {
                            self.trigger_generate(ctx);
                        }
                    });
                });
            });
    }

    /// Full-bleed Wallpaper Canvas screen with floating minimalist studio dock.
    fn draw_wallpaper_showcase(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::BLACK))
            .show(ctx, |ui| {
                let screen_rect = ui.max_rect();

                if let Some(tex) = &self.wallpaper_texture {
                    // Aspect-fill (cover) so the wallpaper fills 100% of the window without black bars
                    let tex_size = tex.size_vec2();
                    let win_aspect = screen_rect.width() / screen_rect.height().max(1.0);
                    let tex_aspect = tex_size.x / tex_size.y.max(1.0);

                    let uv = if win_aspect > tex_aspect {
                        let v_span = tex_aspect / win_aspect;
                        let v0 = (1.0 - v_span) * 0.5;
                        Rect::from_min_max(Pos2::new(0.0, v0), Pos2::new(1.0, v0 + v_span))
                    } else {
                        let u_span = win_aspect / tex_aspect;
                        let u0 = (1.0 - u_span) * 0.5;
                        Rect::from_min_max(Pos2::new(u0, 0.0), Pos2::new(u0 + u_span, 1.0))
                    };

                    ui.painter().image(tex.id(), screen_rect, uv, Color32::WHITE);
                }

                // If HUD is hidden, clicking anywhere restores it
                if self.hide_canvas_hud {
                    let bg_resp = ui.interact(screen_rect, Id::new("clean_canvas_tap"), Sense::click());
                    if bg_resp.clicked() {
                        self.hide_canvas_hud = false;
                    }
                    return;
                }

                // Floating Bottom Glass Control Bar
                egui::Area::new(Id::new("floating_canvas_dock"))
                    .order(Order::Foreground)
                    .anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -24.0))
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(Color32::from_rgba_unmultiplied(14, 15, 22, 212))
                            .stroke(Stroke::new(
                                1.0,
                                Color32::from_rgba_unmultiplied(255, 255, 255, 38),
                            ))
                            .corner_radius(CornerRadius::same(16))
                            .inner_margin(Margin::symmetric(18, 12))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    // Mini active palette dots + style info
                                    for i in 0..self.color_count {
                                        let col = self.slots[i].egui_color();
                                        let (dot_rect, _) =
                                            ui.allocate_exact_size(Vec2::splat(16.0), Sense::hover());
                                        ui.painter().circle(
                                            dot_rect.center(),
                                            7.0,
                                            col,
                                            Stroke::new(
                                                1.0,
                                                Color32::from_rgba_unmultiplied(255, 255, 255, 90),
                                            ),
                                        );
                                    }

                                    if let Some(wp) = &self.current_wallpaper {
                                        ui.add_space(6.0);
                                        ui.label(
                                            RichText::new(format!(
                                                "{}  •  #{:06X}",
                                                wp.style_label,
                                                wp.seed & 0xFF_FFFF
                                            ))
                                            .font(FontId::new(12.5, FontFamily::Proportional))
                                            .color(Color32::from_rgb(178, 184, 204)),
                                        );
                                    }

                                    ui.add_space(12.0);

                                    // Return to Color Picker Studio Menu
                                    let palette_btn = egui::Button::new(
                                        RichText::new("🎨  Ubah Warna")
                                            .font(FontId::new(13.5, FontFamily::Proportional))
                                            .color(Color32::from_rgb(235, 238, 248)),
                                    )
                                    .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 18))
                                    .stroke(Stroke::new(
                                        1.0,
                                        Color32::from_rgba_unmultiplied(255, 255, 255, 32),
                                    ))
                                    .corner_radius(CornerRadius::same(10))
                                    .min_size(Vec2::new(124.0, 38.0));

                                    if ui
                                        .add(palette_btn)
                                        .on_hover_cursor(CursorIcon::PointingHand)
                                        .clicked()
                                    {
                                        self.screen = AppScreen::PaletteMenu;
                                    }

                                    // Generate! again (new random composition & saduran)
                                    let gen_txt = if self.is_generating {
                                        "Melukis..."
                                    } else {
                                        "✦  Generate!"
                                    };
                                    let reroll_btn = egui::Button::new(
                                        RichText::new(gen_txt)
                                            .font(FontId::new(14.5, FontFamily::Proportional))
                                            .color(Color32::from_rgb(14, 16, 22))
                                            .strong(),
                                    )
                                    .fill(Color32::from_rgb(245, 246, 252))
                                    .stroke(Stroke::new(1.0, Color32::WHITE))
                                    .corner_radius(CornerRadius::same(10))
                                    .min_size(Vec2::new(142.0, 38.0));

                                    if ui
                                        .add_enabled(!self.is_generating, reroll_btn)
                                        .on_hover_cursor(CursorIcon::PointingHand)
                                        .on_hover_text("Lukis komposisi & saduran acak baru (Spasi)")
                                        .clicked()
                                    {
                                        self.trigger_generate(ctx);
                                    }

                                    // Save PNG Wallpaper
                                    let save_btn = egui::Button::new(
                                        RichText::new("💾  Simpan PNG")
                                            .font(FontId::new(13.5, FontFamily::Proportional))
                                            .color(Color32::from_rgb(235, 238, 248)),
                                    )
                                    .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 18))
                                    .stroke(Stroke::new(
                                        1.0,
                                        Color32::from_rgba_unmultiplied(255, 255, 255, 32),
                                    ))
                                    .corner_radius(CornerRadius::same(10))
                                    .min_size(Vec2::new(126.0, 38.0));

                                    if ui
                                        .add_enabled(!self.is_exporting, save_btn)
                                        .on_hover_cursor(CursorIcon::PointingHand)
                                        .on_hover_text("Ekspor wallpaper resolusi penuh (2560×1440 PNG)")
                                        .clicked()
                                    {
                                        self.trigger_export_png(ctx);
                                    }

                                    // Clean preview toggle
                                    let hide_btn = egui::Button::new(
                                        RichText::new("👁")
                                            .font(FontId::new(14.0, FontFamily::Proportional))
                                            .color(Color32::from_rgb(210, 215, 232)),
                                    )
                                    .fill(Color32::from_rgba_unmultiplied(255, 255, 255, 12))
                                    .stroke(Stroke::new(
                                        1.0,
                                        Color32::from_rgba_unmultiplied(255, 255, 255, 24),
                                    ))
                                    .corner_radius(CornerRadius::same(10))
                                    .min_size(Vec2::new(38.0, 38.0));

                                    if ui
                                        .add(hide_btn)
                                        .on_hover_cursor(CursorIcon::PointingHand)
                                        .on_hover_text(
                                            "Sembunyikan panel (klik layar atau tekan H untuk kembali)",
                                        )
                                        .clicked()
                                    {
                                        self.hide_canvas_hud = true;
                                    }
                                });
                            });
                    });

                // Status toast notification (e.g. after PNG save)
                let now = ctx.input(|i| i.time);
                if let Some((msg, expiry)) = &self.status_toast {
                    if now < *expiry {
                        egui::Area::new(Id::new("save_toast"))
                            .order(Order::Tooltip)
                            .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 22.0))
                            .show(ctx, |ui| {
                                egui::Frame::NONE
                                    .fill(Color32::from_rgba_unmultiplied(16, 18, 26, 228))
                                    .stroke(Stroke::new(
                                        1.0,
                                        Color32::from_rgba_unmultiplied(255, 255, 255, 45),
                                    ))
                                    .corner_radius(CornerRadius::same(10))
                                    .inner_margin(Margin::symmetric(16, 10))
                                    .show(ui, |ui| {
                                        ui.label(
                                            RichText::new(msg)
                                                .font(FontId::new(13.0, FontFamily::Proportional))
                                                .color(Color32::from_rgb(235, 242, 250)),
                                        );
                                    });
                            });
                        ctx.request_repaint();
                    } else {
                        self.status_toast = None;
                    }
                }
            });
    }
}

/// Draws a single Color Picker Studio Card containing:
/// 1. Live pigment header swatch + editable `#RRGGBB` hex field
/// 2. GPU-rendered 2D Saturation-Value (`SV`) picker canvas
/// 3. GPU-rendered Rainbow Hue (`0°..360°`) slider bar
fn draw_single_color_picker_card(ui: &mut Ui, idx: usize, slot: &mut ColorSlot, card_w: f32) {
    let inner_pad = 14.0_f32;
    let inner_w = (card_w - inner_pad * 2.0).max(160.0);

    egui::Frame::NONE
        .fill(Color32::from_rgba_unmultiplied(24, 26, 36, 232))
        .stroke(Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 26)))
        .corner_radius(CornerRadius::same(16))
        .inner_margin(Margin::same(inner_pad as i8))
        .show(ui, |ui| {
            ui.set_width(inner_w);
            ui.vertical(|ui| {
                // 1. Live Pigment Swatch Header
                let srgb = slot.srgb();
                let swatch_col = Color32::from_rgb(srgb.r, srgb.g, srgb.b);
                let text_on_swatch = if srgb.relative_luminance() > 0.34 {
                    Color32::from_rgba_unmultiplied(15, 17, 24, 230)
                } else {
                    Color32::from_rgba_unmultiplied(248, 250, 255, 242)
                };

                let (swatch_rect, _) =
                    ui.allocate_exact_size(Vec2::new(inner_w, 46.0), Sense::hover());
                ui.painter()
                    .rect_filled(swatch_rect, CornerRadius::same(10), swatch_col);
                ui.painter().rect_stroke(
                    swatch_rect,
                    CornerRadius::same(10),
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 36)),
                    StrokeKind::Inside,
                );

                ui.painter().text(
                    Pos2::new(swatch_rect.left() + 12.0, swatch_rect.center().y),
                    Align2::LEFT_CENTER,
                    format!("WARNA {:02}", idx + 1),
                    FontId::new(12.0, FontFamily::Proportional),
                    text_on_swatch,
                );
                ui.painter().text(
                    Pos2::new(swatch_rect.right() - 12.0, swatch_rect.center().y),
                    Align2::RIGHT_CENTER,
                    srgb.to_hex(),
                    FontId::new(12.5, FontFamily::Monospace),
                    text_on_swatch,
                );

                ui.add_space(10.0);

                // 2. Interactive 2D Saturation-Value (SV) Picker Canvas
                let sv_height = (inner_w * 0.62).clamp(118.0, 168.0);
                let (sv_rect, sv_resp) = ui.allocate_exact_size(
                    Vec2::new(inner_w, sv_height),
                    Sense::click_and_drag(),
                );
                if sv_resp.hovered() || sv_resp.dragged() {
                    ui.ctx().set_cursor_icon(CursorIcon::Crosshair);
                }
                if (sv_resp.clicked() || sv_resp.dragged())
                    && let Some(pos) = sv_resp.interact_pointer_pos()
                {
                    slot.s = ((pos.x - sv_rect.left()) / sv_rect.width()).clamp(0.0, 1.0);
                    slot.v = (1.0 - (pos.y - sv_rect.top()) / sv_rect.height()).clamp(0.0, 1.0);
                    slot.hex_editing = false;
                    slot.sync_hex();
                }

                paint_sv_gradient_mesh(ui, sv_rect, slot.h);
                ui.painter().rect_stroke(
                    sv_rect,
                    CornerRadius::same(4),
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 32)),
                    StrokeKind::Inside,
                );

                // SV Cursor Ring
                let cursor_x = sv_rect.left() + slot.s * sv_rect.width();
                let cursor_y = sv_rect.top() + (1.0 - slot.v) * sv_rect.height();
                let cursor_pos = Pos2::new(cursor_x, cursor_y);
                ui.painter().circle(
                    cursor_pos,
                    7.5,
                    Color32::TRANSPARENT,
                    Stroke::new(2.2, Color32::BLACK),
                );
                ui.painter().circle(
                    cursor_pos,
                    6.5,
                    swatch_col,
                    Stroke::new(2.0, Color32::WHITE),
                );

                ui.add_space(10.0);

                // 3. Interactive Rainbow Hue Bar (0..360 deg)
                let (hue_rect, hue_resp) =
                    ui.allocate_exact_size(Vec2::new(inner_w, 18.0), Sense::click_and_drag());
                if hue_resp.hovered() || hue_resp.dragged() {
                    ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
                }
                if (hue_resp.clicked() || hue_resp.dragged())
                    && let Some(pos) = hue_resp.interact_pointer_pos()
                {
                    let t = ((pos.x - hue_rect.left()) / hue_rect.width()).clamp(0.0, 1.0);
                    // Avoid wrapping 360.0 -> 0.0 when dragging to far right edge
                    slot.h = (t * 359.9).clamp(0.0, 359.9);
                    slot.hex_editing = false;
                    slot.sync_hex();
                }

                paint_hue_spectrum_mesh(ui, hue_rect);
                ui.painter().rect_stroke(
                    hue_rect,
                    CornerRadius::same(4),
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 30)),
                    StrokeKind::Inside,
                );

                // Hue Thumb Handle
                let hx = hue_rect.left() + (slot.h / 360.0) * hue_rect.width();
                let thumb_rect = Rect::from_center_size(
                    Pos2::new(hx, hue_rect.center().y),
                    Vec2::new(8.0, hue_rect.height() + 4.0),
                );
                let pure_hue = Srgb8::from_hsv(slot.h, 1.0, 1.0);
                ui.painter().rect(
                    thumb_rect,
                    CornerRadius::same(4),
                    Color32::from_rgb(pure_hue.r, pure_hue.g, pure_hue.b),
                    Stroke::new(2.0, Color32::WHITE),
                    StrokeKind::Outside,
                );

                ui.add_space(10.0);

                // 4. Direct Hex Code Input Row
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("HEX")
                            .font(FontId::new(11.5, FontFamily::Proportional))
                            .color(Color32::from_rgb(142, 148, 168)),
                    );
                    let edit = egui::TextEdit::singleline(&mut slot.hex_input)
                        .font(FontId::new(13.0, FontFamily::Monospace))
                        .desired_width(inner_w - 42.0)
                        .char_limit(7);
                    let resp = ui.add(edit);
                    if resp.gained_focus() {
                        slot.hex_editing = true;
                    }
                    if resp.changed()
                        && let Some(parsed) = Srgb8::from_hex(&slot.hex_input)
                    {
                        let (h, s, v) = parsed.to_hsv();
                        if s > 1e-4 {
                            slot.h = h;
                        }
                        slot.s = s;
                        slot.v = v;
                    }
                    if resp.lost_focus() {
                        slot.hex_editing = false;
                        slot.sync_hex();
                    }
                });
            });
        });
}

/// Renders an accurate 2D HSV Saturation-Value square for `hue` using an `8x8` GPU vertex mesh.
fn paint_sv_gradient_mesh(ui: &Ui, rect: Rect, hue: f32) {
    const GRID: usize = 8;
    let mut mesh = Mesh::default();

    for gy in 0..=GRID {
        let fy = gy as f32 / GRID as f32;
        let v = 1.0 - fy;
        for gx in 0..=GRID {
            let fx = gx as f32 / GRID as f32;
            let s = fx;
            let c = Srgb8::from_hsv(hue, s, v);
            mesh.vertices.push(egui::epaint::Vertex {
                pos: Pos2::new(rect.left() + fx * rect.width(), rect.top() + fy * rect.height()),
                uv: Pos2::ZERO,
                color: Color32::from_rgb(c.r, c.g, c.b),
            });
        }
    }

    let stride = (GRID + 1) as u32;
    for gy in 0..(GRID as u32) {
        for gx in 0..(GRID as u32) {
            let i0 = gy * stride + gx;
            let i1 = i0 + 1;
            let i2 = i0 + stride;
            let i3 = i2 + 1;
            mesh.add_triangle(i0, i1, i2);
            mesh.add_triangle(i1, i3, i2);
        }
    }

    ui.painter().add(mesh);
}

/// Renders a smooth 0..360 deg Hue spectrum bar on the GPU.
fn paint_hue_spectrum_mesh(ui: &Ui, rect: Rect) {
    const STOPS: usize = 18;
    let mut mesh = Mesh::default();

    for i in 0..=STOPS {
        let t = i as f32 / STOPS as f32;
        let c = Srgb8::from_hsv((t * 360.0).min(359.99), 1.0, 1.0);
        let col = Color32::from_rgb(c.r, c.g, c.b);
        let x = rect.left() + t * rect.width();

        mesh.vertices.push(egui::epaint::Vertex {
            pos: Pos2::new(x, rect.top()),
            uv: Pos2::ZERO,
            color: col,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: Pos2::new(x, rect.bottom()),
            uv: Pos2::ZERO,
            color: col,
        });
    }

    for i in 0..(STOPS as u32) {
        let b = i * 2;
        mesh.add_triangle(b, b + 2, b + 1);
        mesh.add_triangle(b + 1, b + 2, b + 3);
    }

    ui.painter().add(mesh);
}

/// Renders a softly blended horizontal preview ribbon of the active palette slots.
fn paint_pigment_preview_strip(ui: &Ui, rect: Rect, active_slots: &[ColorSlot]) {
    if active_slots.is_empty() {
        return;
    }

    let mut mesh = Mesh::default();
    let n = active_slots.len();
    let samples = n * 6;

    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let scaled = t * (n - 1) as f32;
        let idx0 = (scaled.floor() as usize).min(n - 1);
        let idx1 = (idx0 + 1).min(n - 1);
        let frac = scaled - idx0 as f32;
        // Smoothstep blend along the preview bar
        let s = frac * frac * (3.0 - 2.0 * frac);

        let c0 = active_slots[idx0].srgb();
        let c1 = active_slots[idx1].srgb();
        let r = (c0.r as f32 + (c1.r as f32 - c0.r as f32) * s).round() as u8;
        let g = (c0.g as f32 + (c1.g as f32 - c0.g as f32) * s).round() as u8;
        let b = (c0.b as f32 + (c1.b as f32 - c0.b as f32) * s).round() as u8;
        let col = Color32::from_rgb(r, g, b);

        let x = rect.left() + t * rect.width();
        mesh.vertices.push(egui::epaint::Vertex {
            pos: Pos2::new(x, rect.top()),
            uv: Pos2::ZERO,
            color: col,
        });
        mesh.vertices.push(egui::epaint::Vertex {
            pos: Pos2::new(x, rect.bottom()),
            uv: Pos2::ZERO,
            color: col,
        });
    }

    for i in 0..(samples as u32) {
        let b = i * 2;
        mesh.add_triangle(b, b + 2, b + 1);
        mesh.add_triangle(b + 1, b + 2, b + 3);
    }

    ui.painter().add(mesh);
    ui.painter().rect_stroke(
        rect,
        CornerRadius::same(6),
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 36)),
        StrokeKind::Inside,
    );
}

fn configure_studio_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = Color32::from_rgb(12, 13, 18);
    visuals.window_fill = Color32::from_rgb(18, 20, 28);
    visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(24, 26, 36);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(30, 33, 46);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(42, 46, 62);
    visuals.widgets.active.bg_fill = Color32::from_rgb(56, 62, 82);
    visuals.selection.bg_fill = Color32::from_rgb(82, 118, 212);
    ctx.set_visuals(visuals);
}

fn save_rgba_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| e.to_string())?;
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgba).map_err(|e| e.to_string())?;
    Ok(())
}

