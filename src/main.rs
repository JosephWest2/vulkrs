use app::App;
use winit::event_loop::{ControlFlow, EventLoop};

mod renderer;
mod app;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let mut app = App {
        renderer: None
    };
    let event_loop = EventLoop::new().expect("Failed to create winit event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    _ = event_loop.run_app(&mut app);
}
