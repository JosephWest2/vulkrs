use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop};

use crate::renderer::{Renderer, TriangleVertex};

pub struct App {
    pub renderer: Option<Renderer>,
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
                let vertex_buffer = Buffer::from_iter(
                    renderer.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    VERTICES,
                )
                .unwrap();
                let index_buffer = Buffer::from_iter(
                    renderer.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    INDICES,
                )
                .unwrap();
                renderer.perform_render_pass(vertex_buffer, index_buffer);
                renderer.window.request_redraw();
            }
            _ => (),
        }
    }
}
