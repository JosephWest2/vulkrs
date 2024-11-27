use std::time::{self, Instant};

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::ActiveEventLoop,
};

use crate::{
    camera::{Camera, CameraController},
    renderer::{Renderer, TriangleVertex},
};

pub struct App {
    pub renderer: Option<Renderer>,
    pub camera: Camera,
    pub camera_controller: CameraController,
    pub previous_frame: time::Instant,
}

const VERTICES: [TriangleVertex; 4] = [
    TriangleVertex {
        position: [-0.5, -0.25, 0.0],
    },
    TriangleVertex {
        position: [0.0, 0.5, 0.0],
    },
    TriangleVertex {
        position: [0.25, -0.1, 0.0],
    },
    TriangleVertex {
        position: [0.9, 0.9, 0.0],
    },
];

const INDICES: [u32; 6] = [0, 1, 2, 1, 2, 3];

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.renderer = Some(Renderer::new(event_loop));
        self.renderer.as_mut().unwrap().window.request_redraw();
    }
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.camera_controller.mouse_delta.0 += delta.0 as f32;
                self.camera_controller.mouse_delta.1 += delta.1 as f32;
            }
            _ => (),
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_physical_size) => {
                self.renderer
                    .as_mut()
                    .unwrap()
                    .resize_and_rebuild_swapchain_requested = true;
            }
            WindowEvent::RedrawRequested => {
                let renderer = self.renderer.as_mut().unwrap();
                renderer.perform_render_pass(&self.camera);
                renderer.window.request_redraw();
                let dur = self.previous_frame.elapsed();
                dbg!("{}", dur);
                self.previous_frame = Instant::now();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                use winit::keyboard::{KeyCode, PhysicalKey};
                let is_pressed = event.state.is_pressed();
                match event.physical_key {
                    PhysicalKey::Code(code) => match code {
                        KeyCode::KeyW | KeyCode::ArrowUp => {
                            self.camera_controller.forward_pressed = is_pressed;
                        }
                        KeyCode::KeyS | KeyCode::ArrowDown => {
                            self.camera_controller.backward_pressed = is_pressed;
                        }
                        KeyCode::KeyA | KeyCode::ArrowLeft => {
                            self.camera_controller.left_pressed = is_pressed;
                        }
                        KeyCode::KeyD | KeyCode::ArrowRight => {
                            self.camera_controller.right_pressed = is_pressed;
                        }
                        KeyCode::KeyH => {
                            self.renderer.as_ref().unwrap().update_vertices(&VERTICES.to_vec(), &INDICES.to_vec());
                        }
                        _ => (),
                    },
                    PhysicalKey::Unidentified(_native_key_code) => (),
                }
            }
            _ => (),
        }
    }
}
