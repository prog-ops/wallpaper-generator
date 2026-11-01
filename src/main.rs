#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod color;
pub mod noise;
pub mod painter;
pub mod ui;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Atelier Chroma — Painterly Gradient Wallpaper Generator")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([860.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Atelier Chroma",
        native_options,
        Box::new(|cc| Ok(Box::new(ui::AtelierApp::new(cc)))),
    )
}
