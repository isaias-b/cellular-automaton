use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
    sprite::*,
    transform::commands,
};
use image::{GenericImageView, ImageBuffer};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{path::Path, time::Instant};

const BG_COLOR: Color = Color::rgb(0.0, 0.0, 0.0);
const TILE_SIZE: f32 = 8.0;
const TILE_GAP: f32 = 2.0;
const GRID_DIMENSIONS: (usize, usize) = (256, 256);

// struct Image {
//     width: u32,
//     height: u32,
//     grid: Vec<Vec<Color>>,
// }

#[derive(Component, Debug, Clone, Copy)]
struct Colored(Color);

// impl Image {
//     fn from_path(img_path: &Path) -> Image {
//         let img = image::open(&img_path).expect("Failed to open image");

//         let (width, height) = img.dimensions();

//         let mut grid = vec![vec![Color::BLACK; width as usize]; height as usize];

//         for (x, y, pixel) in img.pixels() {
//             let r = pixel[0] as f32 / 255.0;
//             let g = pixel[1] as f32 / 255.0;
//             let b = pixel[2] as f32 / 255.0;
//             let a = pixel[3] as f32 / 255.0;
//             let color = Color::rgba(r, g, b, a);
//             grid[y as usize][x as usize] = color;
//         }

//         Image {
//             width,
//             height,
//             grid,
//         }
//     }
// }

macro_rules! color_from_hex {
    ($hex:expr) => {
        Color::rgb(
            (($hex >> 16) & 0xFF) as f32 / 255.0,
            (($hex >> 8) & 0xFF) as f32 / 255.0,
            ($hex & 0xFF) as f32 / 255.0,
        )
    };
}

// blend a color onto another color
fn blend_color(top: &Color, bottom: &Color) -> Color {
    let a = top.a() + bottom.a() * (1.0 - top.a());
    let r = top.r() * top.a() + bottom.r() * (1.0 - top.a()) / a;
    let g = top.g() * top.a() + bottom.g() * (1.0 - top.a()) / a;
    let b = top.b() * top.a() + bottom.b() * (1.0 - top.a()) / a;
    Color::rgba(r, g, b, a)
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    color: Color,
    entity: Option<Entity>,
}
impl Cell {
    fn spawn(
        &mut self,
        x: f32,
        y: f32,
        commands: &mut Commands,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
    ) -> Entity {
        let shape = Mesh2dHandle(meshes.add(Rectangle::new(TILE_SIZE, TILE_SIZE)));

        let color = self.color;
        let square = MaterialMesh2dBundle {
            mesh: shape,
            material: materials.add(color),
            transform: Transform::from_xyz(x, y, 0.0),
            ..default()
        };

        let entity = commands.spawn(square).insert(Colored(color)).id();
        self.entity = Some(entity);
        entity
    }
}

#[derive(Resource)]
struct Grid {
    width: usize,
    height: usize,
    cells: Vec<Vec<Cell>>,
    entity: Option<Entity>,
}

struct Center {
    x: usize,
    y: usize,
}
struct Kernel {
    width: usize,
    height: usize,
    cells: Vec<Vec<f32>>,
}
impl Kernel {
    fn gauss() -> Kernel {
        let cells = vec![
            vec![1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
            vec![2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
            vec![1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        ];
        Kernel {
            width: 3,
            height: 3,
            cells,
        }
    }

    fn center(&self) -> Center {
        Center {
            x: self.width / 2,
            y: self.height / 2,
        }
    }
}

type Projection = dyn Fn(&Color, &mut Cell);
struct RGBA {
    r: f32,
    g: f32,
    b: f32,
}

impl Grid {
    // fn update_with_image(&mut self, image: &Image, projection: &Projection) {
    //     for (y, row) in self.cells.iter_mut().enumerate() {
    //         for (x, cell) in row.iter_mut().enumerate() {
    //             let img_in_color = &image.grid[y][x];
    //             projection(img_in_color, cell);
    //         }
    //     }
    // }
    fn convolve(&mut self, kernel: &Kernel) {
        let mut new_cells = self.cells.clone();
        let kc = kernel.center();
        let mut new_color;
        for y in 0..self.height {
            for x in 0..self.width {
                new_color = RGBA {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                };
                for ky in 0..kernel.height {
                    for kx in 0..kernel.width {
                        let dx = x as i32 + kx as i32 - kc.x as i32;
                        let dy = y as i32 + ky as i32 - kc.y as i32;
                        if dx < 0 || dy < 0 {
                            continue;
                        }
                        if dx >= self.width as i32 || dy >= self.height as i32 {
                            continue;
                        }
                        let cell = &self.cells[dy as usize][dx as usize];
                        let weight = kernel.cells[ky][kx];
                        new_color.r += cell.color.r() * weight;
                        new_color.g += cell.color.g() * weight;
                        new_color.b += cell.color.b() * weight;
                    }
                }
                new_cells[y][x].color = Color::rgba(new_color.r, new_color.g, new_color.b, 1.0);
            }
        }
        self.cells = new_cells;
    }
    fn new_random(width: usize, height: usize) -> Grid {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut cells = vec![
            vec![
                Cell {
                    color: Color::BLACK,
                    entity: None,
                };
                width
            ];
            height
        ];
        for row in cells.iter_mut() {
            for cell in row.iter_mut() {
                let r = rng.gen_range(0..255) as f32 / 255.0;
                let g = rng.gen_range(0..255) as f32 / 255.0;
                let b = rng.gen_range(0..255) as f32 / 255.0;
                cell.color = Color::rgb(r, g, b);
            }
        }

        Grid {
            width,
            height,
            cells,
            entity: None,
        }
    }
    // fn new_from_image(image: &Image, projection: &Projection) -> Grid {
    //     let cells = vec![
    //         vec![
    //             Cell {
    //                 color: Color::BLACK,
    //                 entity: None,
    //             };
    //             image.width as usize
    //         ];
    //         image.height as usize
    //     ];

    //     let mut world = Grid {
    //         width: image.width as usize,
    //         height: image.height as usize,
    //         cells,
    //     };
    //     world.update_with_image(image, projection);
    //     world
    // }

    fn spawn_ui(
        &mut self,
        commands: &mut Commands,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
    ) {
        let len = self.cells.len();
        let xcenter = (len as f32 * TILE_SIZE + (len - 1) as f32 * TILE_GAP) / 2.0;
        let ycenter = (len as f32 * TILE_SIZE + (len - 1) as f32 * TILE_GAP) / 2.0;
        for (y, row) in self.cells.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                let xpos = x as f32 * (TILE_SIZE + TILE_GAP);
                let ypos = (len - y as usize) as f32 * (TILE_SIZE + TILE_GAP);
                cell.spawn(xpos - xcenter, ypos - ycenter, commands, meshes, materials);
            }
        }
    }
    fn spawn_ui_as_texture(
        &mut self,
        commands: &mut Commands,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
        images: &mut ResMut<Assets<Image>>,
    ) {
        let texture: Vec<u8> = self.as_large_texture();
        let texture_handle = images.add(Image::new(
            Extent3d {
                width: self.width as u32 * TILE_SIZE as u32,
                height: self.height as u32 * TILE_SIZE as u32,
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
            self.width as f32 * TILE_SIZE,
            self.height as f32 * TILE_SIZE,
        )));
        let square = MaterialMesh2dBundle {
            mesh: shape,
            material: material_handle,
            transform: Transform::from_translation(Vec3::ZERO),
            ..default()
        };
        self.entity = Some(commands.spawn(square).id());
    }
    fn as_large_texture(&mut self) -> Vec<u8> {
        let pwidth = self.width * TILE_SIZE as usize;
        let pheight = self.height * TILE_SIZE as usize;
        let mut imgbuf = ImageBuffer::new(pwidth as u32, pheight as u32);
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = &self.cells[y][x];
                let color = cell.color;
                for ty in 0..TILE_SIZE as u32 {
                    for tx in 0..TILE_SIZE as u32 {
                        imgbuf.put_pixel(
                            x as u32 * TILE_SIZE as u32 + tx,
                            y as u32 * TILE_SIZE as u32 + ty,
                            image::Rgba([
                                (color.r() * 255.0) as u8,
                                (color.g() * 255.0) as u8,
                                (color.b() * 255.0) as u8,
                                (color.a() * 255.0) as u8,
                            ]),
                        );
                    }
                }
            }
        }
        imgbuf.into_raw()
    }

    fn update_ui_as_texture(
        &mut self,
        commands: &mut Commands,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<ColorMaterial>>,
        images: &mut ResMut<Assets<Image>>,
    ) {
        let texture: Vec<u8> = self.as_large_texture();
        let texture_handle = images.add(Image::new(
            Extent3d {
                width: self.width as u32 * TILE_SIZE as u32,
                height: self.height as u32 * TILE_SIZE as u32,
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
            self.width as f32 * TILE_SIZE,
            self.height as f32 * TILE_SIZE,
        )));
        let square = MaterialMesh2dBundle {
            mesh: shape,
            material: material_handle,
            transform: Transform::from_translation(Vec3::ZERO),
            ..default()
        };

        let entity = self.entity.unwrap();
        commands.entity(entity).insert(square);
    }

    fn update_ui(&mut self, commands: &mut Commands) {
        for row in self.cells.iter_mut() {
            for cell in row.iter_mut() {
                commands
                    .entity(cell.entity.unwrap())
                    .insert(Colored(cell.color));
            }
        }
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
    let mut world = Grid::new_random(GRID_DIMENSIONS.0, GRID_DIMENSIONS.1);

    // world.spawn_ui(&mut commands, &mut meshes, &mut materials);
    world.spawn_ui_as_texture(&mut commands, &mut meshes, &mut materials, &mut images);
    commands.insert_resource(world);
}

// fn update_colored(
//     query: Query<(&Handle<ColorMaterial>, &Colored), Changed<Colored>>,
//     mut materials: ResMut<Assets<ColorMaterial>>,
// ) {
//     let time = Instant::now();
//     println!(
//         "{:?}: Updating colored entities {:?}",
//         time,
//         query.iter().count()
//     );
//     if (query.iter().count() == 0) {
//         return;
//     }
//     for (material, colored) in query.iter() {
//         let color = colored.0;
//         let material = materials.get_mut(material).unwrap();
//         material.color = color;
//     }
//     println!("Updating colored entities took {:?}", time.elapsed());
// }

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut world: ResMut<Grid>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        let kernel = Kernel::gauss();
        let time = Instant::now();
        world.convolve(&kernel);
        println!("Convolution took {:?}", time.elapsed());
        world.update_ui_as_texture(&mut commands, &mut meshes, &mut materials, &mut images);
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
        // .add_systems(Update, update_colored)
        .add_systems(Update, measure_frame_time)
        .add_systems(Startup, setup)
        .run();
}
