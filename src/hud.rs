// egui overlay: stats panel, hotbar, crosshair, control hints.

use crate::world::{block, HOTBAR};
use egui::{Align2, Color32, FontId, Rgba, Stroke, Vec2};

pub struct Payload {
    pub fps: f32,
    pub pos: [f32; 3],
    pub chunks: usize,
    pub queue: usize,
    pub hotbar_sel: usize,
    pub fly: bool,
    pub locked: bool,
}

pub fn paint(ctx: &egui::Context, p: &Payload) {
    // ---- stats panel ----
    egui::Window::new("stats")
        .fixed_pos(egui::pos2(10.0, 10.0))
        .title_bar(false)
        .frame(egui::Frame::NONE.fill(Color32::from_black_alpha(120)).corner_radius(6.0))
        .show(ctx, |ui| {
            ui.monospace(format!("{:5.0} fps", p.fps));
            ui.monospace(format!("xyz {:7.1} {:7.1} {:7.1}", p.pos[0], p.pos[1], p.pos[2]));
            ui.monospace(format!("chunks {:>4}  queue {:>4}", p.chunks, p.queue));
            if p.fly {
                ui.colored_label(Color32::from_rgb(120, 220, 255), "FLY");
            }
        });

    // ---- hotbar ----
    let screen = ctx.content_rect();
    egui::Area::new(egui::Id::new("hotbar"))
        .fixed_pos(egui::pos2(screen.center().x, screen.max.y - 20.0))
        .pivot(egui::Align2::CENTER_BOTTOM)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(4.0, 0.0);
                for (i, b) in HOTBAR.iter().enumerate() {
                    let sel = i == p.hotbar_sel;
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(44.0), egui::Sense::hover());
                    let painter = ui.painter();
                    let frame = if sel {
                        Stroke::new(3.0_f32, Color32::from_rgb(255, 255, 255))
                    } else {
                        Stroke::new(1.5_f32, Color32::from_gray(140))
                    };
                    painter.rect_filled(rect, 4.0, Color32::from_black_alpha(150));
                    painter.rect_stroke(rect, 4.0, frame, egui::StrokeKind::Inside);
                    // swatch colored like the block (matches atlas bases)
                    let sw = rect.shrink(10.0);
                    painter.rect_filled(sw, 2.0, block_swatch(*b));
                    painter.text(
                        rect.left_top() + Vec2::new(3.0, -1.0),
                        Align2::LEFT_BOTTOM,
                        (i + 1).to_string(),
                        FontId::proportional(12.0),
                        Color32::from_gray(200),
                    );
                }
            });
        });

    // ---- crosshair ----
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("xhair")));
    let c = screen.center();
    let stroke = Stroke::new(2.0_f32, Color32::from_white_alpha(220));
    painter.line_segment([c + Vec2::new(-9.0, 0.0), c + Vec2::new(-3.0, 0.0)], stroke);
    painter.line_segment([c + Vec2::new(3.0, 0.0), c + Vec2::new(9.0, 0.0)], stroke);
    painter.line_segment([c + Vec2::new(0.0, -9.0), c + Vec2::new(0.0, -3.0)], stroke);
    painter.line_segment([c + Vec2::new(0.0, 3.0), c + Vec2::new(0.0, 9.0)], stroke);

    // ---- selected block name ----
    painter.text(
        c + Vec2::new(0.0, 44.0),
        Align2::CENTER_TOP,
        block::name(HOTBAR[p.hotbar_sel]),
        FontId::proportional(15.0),
        Color32::from_white_alpha(200),
    );

    // ---- unlock hint / menu ----
    if !p.locked {
        egui::Window::new("menu")
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .title_bar(false)
            .frame(egui::Frame::NONE.fill(Color32::from_black_alpha(170)).corner_radius(8.0))
            .show(ctx, |ui| {
                ui.heading("Minecraft-Clone");
                ui.label("Click the window to capture the mouse");
                ui.separator();
                ui.monospace("WASD     move        Space   jump/up");
                ui.monospace("Shift    sneak/down  Ctrl    sprint");
                ui.monospace("F        toggle fly  1-9     block");
                ui.monospace("Wheel    hotbar      LMB/RMB break/place");
                ui.monospace("Esc      release mouse");
            });
    }
}

/// 2D swatch tint per block id (approximates the atlas tile).
fn block_swatch(b: u8) -> Rgba {
    use crate::world::block::*;
    let c: [u8; 3] = match b {
        GRASS => [96, 168, 64],
        DIRT => [134, 96, 67],
        STONE => [128, 128, 128],
        COBBLE => [118, 118, 118],
        SAND => [219, 207, 160],
        WATER => [48, 106, 190],
        LOG => [104, 82, 50],
        LEAVES => [54, 128, 44],
        GLASS => [176, 216, 228],
        SNOW => [240, 246, 250],
        PLANKS => [168, 132, 82],
        BEDROCK => [60, 60, 66],
        GRAVEL => [130, 126, 124],
        _ => [0, 0, 0],
    };
    Rgba::from_rgb(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0)
}
