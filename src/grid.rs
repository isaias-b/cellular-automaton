use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};
use std::{collections::HashMap, time::Instant};

#[derive(Clone, Copy, Debug)]
pub struct RGBA {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub color: RGBA,
}
pub struct Center {
    pub x: usize,
    pub y: usize,
}

#[derive(Clone)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Cell>,
}

pub struct Kernel {
    width: usize,
    height: usize,
    cells: Vec<Vec<f32>>,
}

impl Kernel {
    pub fn gauss3() -> Kernel {
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
    pub fn gauss5() -> Kernel {
        #[rustfmt::skip]
      let cells = vec![
          vec![1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0],
          vec![4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0],
          vec![6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0],
          vec![4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0],
          vec![1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0],
      ];
        Kernel {
            width: 5,
            height: 5,
            cells,
        }
    }
    pub fn gauss7() -> Kernel {
        #[rustfmt::skip]
      let cells = vec![
          vec![1.0 / 4096.0, 6.0 / 4096.0, 15.0 / 4096.0, 20.0 / 4096.0, 15.0 / 4096.0, 6.0 / 4096.0, 1.0 / 4096.0],
          vec![6.0 / 4096.0, 36.0 / 4096.0, 90.0 / 4096.0, 120.0 / 4096.0, 90.0 / 4096.0, 36.0 / 4096.0, 6.0 / 4096.0],
          vec![15.0 / 4096.0, 90.0 / 4096.0, 225.0 / 4096.0, 300.0 / 4096.0, 225.0 / 4096.0, 90.0 / 4096.0, 15.0 / 4096.0],
          vec![20.0 / 4096.0, 120.0 / 4096.0, 300.0 / 4096.0, 400.0 / 4096.0, 300.0 / 4096.0, 120.0 / 4096.0, 20.0 / 4096.0],
          vec![15.0 / 4096.0, 90.0 / 4096.0, 225.0 / 4096.0, 300.0 / 4096.0, 225.0 / 4096.0, 90.0 / 4096.0, 15.0 / 4096.0],
          vec![6.0 / 4096.0, 36.0 / 4096.0, 90.0 / 4096.0, 120.0 / 4096.0, 90.0 / 4096.0, 36.0 / 4096.0, 6.0 / 4096.0],
          vec![1.0 / 4096.0, 6.0 / 4096.0, 15.0 / 4096.0, 20.0 / 4096.0, 15.0 / 4096.0, 6.0 / 4096.0, 1.0 / 4096.0],
      ];
        Kernel {
            width: 7,
            height: 7,
            cells,
        }
    }

    pub fn center(&self) -> Center {
        Center {
            x: self.width / 2,
            y: self.height / 2,
        }
    }
}

impl Grid {
    #[inline(always)]
    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> &Cell {
        &self.cells[self.index(x, y)]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut Cell {
        let index = self.index(x, y);
        &mut self.cells[index]
    }

    pub fn convolve_fft(&mut self, kernel: &Kernel) {
        let (grid_width, grid_height) = (self.width, self.height);
        let (kernel_width, kernel_height) = (kernel.width, kernel.height);
        let padded_width = grid_width + kernel_width - 1;
        let padded_height = grid_height + kernel_height - 1;
        let time = Instant::now();
        let mut planner = FftPlanner::new();
        println!("Planner took {:?}", time.elapsed());
        let time = Instant::now();
        let fft = planner.plan_fft_forward(padded_width * padded_height);
        println!("FFT forward took {:?}", time.elapsed());
        let time = Instant::now();
        let ifft = planner.plan_fft_inverse(padded_width * padded_height);
        println!("FFT inverse took {:?}", time.elapsed());

        let time = Instant::now();
        let mut grid_r = vec![Complex::zero(); padded_width * padded_height];
        let mut grid_g = vec![Complex::zero(); padded_width * padded_height];
        let mut grid_b = vec![Complex::zero(); padded_width * padded_height];
        let mut grid_a = vec![Complex::zero(); padded_width * padded_height];
        let mut kernel_complex = vec![Complex::zero(); padded_width * padded_height];

        for y in 0..grid_height {
            for x in 0..grid_width {
                let index = y * padded_width + x;
                let color = self.get(x, y).color;
                grid_r[index] = Complex::new(color.r, 0.0);
                grid_g[index] = Complex::new(color.g, 0.0);
                grid_b[index] = Complex::new(color.b, 0.0);
                grid_a[index] = Complex::new(color.a, 0.0);
            }
        }

        for y in 0..kernel_height {
            for x in 0..kernel_width {
                let index = y * padded_width + x;
                kernel_complex[index] = Complex::new(kernel.cells[y][x], 0.0);
            }
        }
        println!("Preparation took {:?}", time.elapsed());
        let time = Instant::now();
        fft.process(&mut grid_r);
        fft.process(&mut grid_g);
        fft.process(&mut grid_b);
        fft.process(&mut grid_a);
        fft.process(&mut kernel_complex);
        println!("FFT processing took {:?}", time.elapsed());

        for i in 0..padded_width * padded_height {
            grid_r[i] *= kernel_complex[i];
            grid_g[i] *= kernel_complex[i];
            grid_b[i] *= kernel_complex[i];
            grid_a[i] *= kernel_complex[i];
        }

        let time = Instant::now();
        ifft.process(&mut grid_r);
        ifft.process(&mut grid_g);
        ifft.process(&mut grid_b);
        ifft.process(&mut grid_a);
        println!("IFFT processing took {:?}", time.elapsed());

        let kernel_center = kernel.center();
        for y in 0..grid_height {
            for x in 0..grid_width {
                let dx = x as i32 + kernel_center.x as i32;
                let dy = y as i32 + kernel_center.y as i32;
                let index = dy as usize * padded_width + dx as usize;
                let r_value = grid_r[index].re / (padded_width * padded_height) as f32;
                let g_value = grid_g[index].re / (padded_width * padded_height) as f32;
                let b_value = grid_b[index].re / (padded_width * padded_height) as f32;
                let a_value = grid_a[index].re / (padded_width * padded_height) as f32;

                let cell = self.get_mut(x, y);
                cell.color.r = r_value.clamp(0.0, 1.0);
                cell.color.g = g_value.clamp(0.0, 1.0);
                cell.color.b = b_value.clamp(0.0, 1.0);
                cell.color.a = a_value.clamp(0.0, 1.0);
            }
        }
    }

    pub fn convolve_par(&mut self, kernel: &Kernel) {
        let kc = kernel.center();

        let result: HashMap<usize, Cell> = (0..self.width * self.height)
            .into_par_iter()
            .fold(HashMap::new, |mut acc, index| {
                let y = index / self.width;
                let x = index % self.width;
                let mut new_color = RGBA {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
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
                        let cell = &self.get(dx as usize, dy as usize);
                        let weight = kernel.cells[ky][kx];
                        new_color.r += cell.color.r * weight;
                        new_color.g += cell.color.g * weight;
                        new_color.b += cell.color.b * weight;
                        new_color.a += cell.color.a * weight;
                    }
                }
                let new_color = RGBA {
                    r: new_color.r.clamp(0.0, 1.0),
                    g: new_color.g.clamp(0.0, 1.0),
                    b: new_color.b.clamp(0.0, 1.0),
                    a: new_color.a.clamp(0.0, 1.0),
                };
                acc.insert(index, Cell { color: new_color });
                acc
            })
            .reduce(HashMap::new, |mut acc, map| {
                acc.extend(map);
                acc
            });

        for (index, cell) in result {
            self.cells[index] = cell;
        }
    }

    pub fn convolve(&mut self, kernel: &Kernel) {
        let mut new_cells = self.cells.clone();
        let kc = kernel.center();

        let mut new_color;
        for y in 0..self.height {
            for x in 0..self.width {
                new_color = RGBA {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
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
                        let cell = &self.get(dx as usize, dy as usize);
                        let weight = kernel.cells[ky][kx];
                        new_color.r += cell.color.r * weight;
                        new_color.g += cell.color.g * weight;
                        new_color.b += cell.color.b * weight;
                    }
                }
                let index = self.index(x, y);
                new_cells[index].color = new_color;
            }
        }
        self.cells = new_cells;
    }

    pub fn center(&self) -> Center {
        Center {
            x: self.width / 2,
            y: self.height / 2,
        }
    }

    pub fn new_random(width: usize, height: usize) -> Grid {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut cells = vec![
            Cell {
                color: RGBA {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                }
            };
            width * height
        ];

        for cell in cells.iter_mut() {
            let r = rng.gen_range(0..255) as f32 / 255.0;
            let g = rng.gen_range(0..255) as f32 / 255.0;
            let b = rng.gen_range(0..255) as f32 / 255.0;
            cell.color = RGBA { r, g, b, a: 1.0 };
        }

        Grid {
            width,
            height,
            cells,
        }
    }
}
