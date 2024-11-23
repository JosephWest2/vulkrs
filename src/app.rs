use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop};

use crate::renderer::Renderer;

pub struct App {
    pub renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.renderer = Some(Renderer::new(event_loop));
        self.renderer.as_mut().unwrap().window.request_redraw();
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_physical_size) => {
                self.renderer.as_mut().unwrap().resize_requested = true;
            }
            WindowEvent::RedrawRequested => {
                self.renderer.as_mut().unwrap().perform_render_pass();
                self.renderer.as_mut().unwrap().window.request_redraw();
            }
            _ => (),
        }
    }
}
