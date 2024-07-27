#[macro_use]
mod grid;
mod kernel;
mod rgba;

use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
    sprite::*,
};
use image::ImageBuffer;
use kernel::*;
use rgba::*;
use std::time::Instant;

use grid::*;

const BG_COLOR: Color = Color::rgb(0.0, 0.0, 0.0);
const TILE_SIZE: f32 = 8.0;
const GRID_DIMENSIONS: (usize, usize) = (512, 512);

#[derive(Resource)]
struct World {
    grid: Grid<RGBA>,
    entity: Option<Entity>,
}

fn create_grid_texture(
    grid: &Grid<RGBA>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    images: &mut ResMut<Assets<Image>>,
) -> MaterialMesh2dBundle<ColorMaterial> {
    let pwidth = grid.width() * TILE_SIZE as usize;
    let pheight = grid.height() * TILE_SIZE as usize;
    let mut imgbuf = ImageBuffer::new(pwidth as u32, pheight as u32);
    for y in 0..grid.height() {
        for x in 0..grid.width() {
            let cell = &grid.get(x, y);
            for ty in 0..TILE_SIZE as u32 {
                for tx in 0..TILE_SIZE as u32 {
                    imgbuf.put_pixel(
                        x as u32 * TILE_SIZE as u32 + tx,
                        y as u32 * TILE_SIZE as u32 + ty,
                        image::Rgba([
                            (cell.r * 255.0) as u8,
                            (cell.g * 255.0) as u8,
                            (cell.b * 255.0) as u8,
                            (cell.a * 255.0) as u8,
                        ]),
                    );
                }
            }
        }
    }
    let texture: Vec<u8> = imgbuf.into_raw();
    let texture_handle = images.add(Image::new(
        Extent3d {
            width: grid.width() as u32 * TILE_SIZE as u32,
            height: grid.height() as u32 * TILE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        texture,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    ));
    let material_handle = materials.add(ColorMaterial {
        texture: Some(texture_handle.clone()),
        ..Default::default()
    });
    let shape = Mesh2dHandle(meshes.add(Rectangle::new(
        grid.width() as f32 * TILE_SIZE,
        grid.height() as f32 * TILE_SIZE,
    )));
    MaterialMesh2dBundle {
        mesh: shape,
        material: material_handle,
        transform: Transform::from_translation(Vec3::ZERO),
        ..default()
    }
}
fn measure_frame_time(mut last_time: Local<Option<Instant>>) {
    let now = Instant::now();
    if let Some(last_time) = *last_time {
        println!("Frame time: {:?}", now.duration_since(last_time));
    }
    *last_time = Some(now);
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.spawn(Camera2dBundle::default());
    let grid = Grid::new_random(GRID_DIMENSIONS.0, GRID_DIMENSIONS.1);
    println!(
        "Grid size: {}x{} = {} cells",
        grid.width(),
        grid.height(),
        grid.width() * grid.height()
    );
    let square = create_grid_texture(&grid, &mut meshes, &mut materials, &mut images);
    let entity = commands.spawn(square).id();
    commands.insert_resource(World {
        grid: grid,
        entity: Some(entity),
    });
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut world: ResMut<World>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        let kernel = Kernel::gauss7();
        world.grid.convolve(&kernel, ParConvolver);
        let square = create_grid_texture(&world.grid, &mut meshes, &mut materials, &mut images);
        let entity = world.entity.unwrap();
        commands.entity(entity).insert(square);
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            file_path: "assets".to_string(),
            ..default()
        }))
        .insert_resource(ClearColor(BG_COLOR))
        .add_systems(Update, bevy::window::close_on_esc)
        .add_systems(Update, handle_input)
        // .add_systems(Update, measure_frame_time)
        .add_systems(Startup, setup)
        .run();
}
