use app::App;
use camera::{Camera, CameraController};
use winit::event_loop::{ControlFlow, EventLoop};

mod renderer;
mod app;
mod camera;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let mut app = App {
        renderer: None,
        camera: Camera::new(1920.0/1080.0),
        camera_controller: CameraController::new(),
    };
    let event_loop = EventLoop::new().expect("Failed to create winit event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    _ = event_loop.run_app(&mut app);
}
