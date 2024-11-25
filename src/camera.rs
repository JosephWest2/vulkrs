use std::f32::consts::PI;

use nalgebra::{Matrix4, Point3, Vector3};
use vulkano::buffer::BufferContents;

pub struct Camera {
    pub position: Point3<f32>,
    pub pitch: f32,
    pub yaw: f32,
    up: Vector3<f32>,
    // aspect ratio equals screen width / screen height
    aspect_ratio: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct CameraUniform {
    pub view_projection_matrix: [[f32; 4]; 4],
}

impl Camera {
    pub fn new(aspect_ratio: f32) -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
            up: Vector3::y_axis().into_inner(),
            aspect_ratio,
            fovy: PI / 4.0,
            znear: 0.01,
            zfar: 100.0,
        }
    }
    pub fn get_camera_forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            -1.0 * self.pitch.cos() * self.yaw.cos(),
        )
    }
    pub fn get_view_projection_matrix(&self) -> Matrix4<f32> {
        let view_matrix = Matrix4::look_at_rh(
            &self.position,
            &(self.position + self.get_camera_forward()),
            &self.up,
        );
        let projection = nalgebra::Perspective3::new(
            self.aspect_ratio,
            self.fovy.to_degrees(),
            self.znear,
            self.zfar,
        );
        return projection.as_matrix() * view_matrix;
    }
    pub fn get_uniform(&self) -> CameraUniform {
        CameraUniform {
            view_projection_matrix: self.get_view_projection_matrix().into(),
        }
    }
}

pub struct CameraController {
    pub movement_speed: f32,
    pub mouse_sens: f32,
    pub mouse_delta: (f32, f32),
    pub forward_pressed: bool,
    pub backward_pressed: bool,
    pub left_pressed: bool,
    pub right_pressed: bool,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            movement_speed: 0.01,
            mouse_sens: 0.01,
            mouse_delta: (0.0, 0.0),
            forward_pressed: false,
            backward_pressed: false,
            left_pressed: false,
            right_pressed: false,
        }
    }
    pub fn update_camera(&mut self, camera: &mut Camera) {
        let forward = camera.get_camera_forward();
        let right = forward.cross(&Vector3::y_axis());
        if self.forward_pressed {
            camera.position += forward * self.movement_speed;
        }
        if self.backward_pressed {
            camera.position -= forward * self.movement_speed;
        }
        if self.left_pressed {
            camera.position -= right * self.movement_speed;
        }
        if self.right_pressed {
            camera.position += right * self.movement_speed;
        }
        camera.yaw += self.mouse_delta.0 * self.mouse_sens;
        camera.pitch -= self.mouse_delta.1 * self.mouse_sens;
        self.mouse_delta = (0.0, 0.0);
    }
}
