use std::{collections::BTreeMap, sync::Arc};

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateFlags, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfoTyped, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo},
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags,
            DescriptorSetLayoutCreateInfo, DescriptorType,
        },
        DescriptorBindingResources, DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        Queue, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, DeviceLayout, FreeListAllocator, GenericMemoryAllocator,
        MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::{
            PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateFlags,
            PipelineLayoutCreateInfo,
        },
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::DescriptorBindingRequirements,
    swapchain::{
        acquire_next_image, PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo
    },
    sync::GpuFuture,
    DeviceSize, Validated, VulkanError, VulkanLibrary,
};
use winit::{event_loop::ActiveEventLoop, window::Window};

use crate::camera::{Camera, CameraUniform};

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct TriangleVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

pub struct Renderer {
    pub window: Arc<Window>,
    pub resize_and_rebuild_swapchain_requested: bool,
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pipeline: Arc<GraphicsPipeline>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    viewport: Viewport,
    vertex_buffer: Subbuffer<[TriangleVertex]>,
    index_buffer: Subbuffer<[u32]>,
    uniform_buffer_allocator: SubbufferAllocator,
}

impl Renderer {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .expect("Failed to create winit window"),
        );
        let vulkan_library =
            VulkanLibrary::new().expect("No default vulkan library found on this system");
        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Instance::new(
            vulkan_library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("Failed to create vulkano instance");
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let surface = Surface::from_window(instance.clone(), window.clone())
            .expect("Failed to create vulkano surface");
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|physical_device| {
                physical_device
                    .supported_extensions()
                    .contains(&device_extensions)
            })
            .filter_map(|physical_device| {
                physical_device
                    .queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, queue_family_properties)| {
                        queue_family_properties
                            .queue_flags
                            .intersects(QueueFlags::GRAPHICS)
                            && physical_device.surface_support(i as u32, &surface).unwrap()
                    })
                    .map(|i| (physical_device, i as u32))
            })
            .min_by_key(
                |(physical_device, _)| match physical_device.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                },
            )
            .unwrap();

        eprintln!(
            "Using device: {} of type {:?}",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, _color_space) = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    present_mode: PresentMode::Immediate,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        mod vertex_shader {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "shaders/vertex_shader.glsl",
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "shaders/fragment_shader.glsl",
            }
        }

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        let vertex_buffer = Buffer::new_unsized::<[TriangleVertex]>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            128,
        )
        .unwrap();

        let index_buffer = Buffer::new_unsized::<[u32]>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            128,
        )
        .unwrap();

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let pipeline = {
            let vertex_shader = vertex_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fragment_shader = fragment_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = TriangleVertex::per_vertex()
                .definition(&vertex_shader.info().input_interface)
                .unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vertex_shader),
                PipelineShaderStageCreateInfo::new(fragment_shader),
            ];
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    depth_stencil_state: None,
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };


        let previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());

        let framebuffers = Self::new_framebuffers(&images, &render_pass);

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        Self {
            window,
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            pipeline,
            render_pass,
            previous_frame_end,
            framebuffers,
            swapchain,
            viewport,
            vertex_buffer,
            index_buffer,
            uniform_buffer_allocator,
            descriptor_set_allocator,
            resize_and_rebuild_swapchain_requested: false,
        }
    }

    fn new_framebuffers(
        images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn resize_and_rebuild_swapchain(&mut self) {
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: self.window.inner_size().into(),
                ..self.swapchain.create_info()
            })
            .expect("Failed to recreate swapchain");
        self.swapchain = new_swapchain;
        self.framebuffers = Self::new_framebuffers(&new_images, &self.render_pass);
        self.viewport.extent = self.window.inner_size().into();
    }

    pub fn update_vertices(&self, vertices: &Vec<TriangleVertex>, indices: &Vec<u32>) {
        if indices.len() * std::mem::size_of::<u32>() > self.index_buffer.size() as usize {
            eprintln!("INDEX BUFFER SMALLER THAN INDEX INPUT");
        }
        if vertices.len() * std::mem::size_of::<TriangleVertex>()
            > self.vertex_buffer.size() as usize
        {
            eprintln!("VERTEX BUFFER SMALLER THAN VERTEX INPUT");
        }
        let staging_vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices.to_vec(),
        )
        .unwrap();
        let staging_index_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices.to_vec(),
        )
        .unwrap();

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer_builder
            .copy_buffer(CopyBufferInfoTyped::buffers(
                staging_vertex_buffer,
                self.vertex_buffer.clone(),
            ))
            .unwrap()
            .copy_buffer(CopyBufferInfoTyped::buffers(
                staging_index_buffer,
                self.index_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();

        //BLOCKS!
        command_buffer
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn perform_render_pass(&mut self, camera: &Camera) {
        let window_size = self.window.inner_size();
        if window_size.width < 1 || window_size.height < 1 {
            return;
        }

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.resize_and_rebuild_swapchain_requested {
            self.resize_and_rebuild_swapchain();
            self.resize_and_rebuild_swapchain_requested = false;
        }

        //This call blocks if there is no available image from the swapchain!
        let (image_index, suboptimal, aquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.resize_and_rebuild_swapchain_requested = true;
                    return;
                }
                Err(e) => panic!("Failed to aquire next image from swapchain: {e}"),
            };


        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        if suboptimal {
            self.resize_and_rebuild_swapchain_requested = true;
        }

        let camera_uniform_buffer = self.uniform_buffer_allocator
            .allocate_sized::<CameraUniform>()
            .unwrap();
        *camera_uniform_buffer.write().unwrap() = camera.get_uniform();
        let camera_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, camera_uniform_buffer)],
            [],
        )
        .unwrap();

        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
            .bind_index_buffer(self.index_buffer.clone())
            .unwrap()
            .bind_descriptor_sets(
                self.pipeline.bind_point(),
                self.pipeline.layout().clone(),
                0,
                vec![camera_descriptor_set.clone().offsets([])],
            )
            .unwrap();

        command_buffer_builder
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();

        command_buffer_builder
            .end_render_pass(Default::default())
            .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(aquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.resize_and_rebuild_swapchain_requested = true;
                self.previous_frame_end = Some(vulkano::sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                panic!("Failed to flush future: {e}");
            }
        }
    }
}
