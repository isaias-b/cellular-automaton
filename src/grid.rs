use crate::kernel::Kernel;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct RGBA {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}
impl RGBA {
    const ZERO: RGBA = RGBA {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
}

pub struct Center {
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
}

impl Raster {
    #[inline(always)]
    pub fn for_each(&self, mut f: impl FnMut(usize, usize, usize)) {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = self.index(x, y);
                f(x, y, index);
            }
        }
    }
    #[inline(always)]
    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline(always)]
    pub fn is_inside(&self, x: isize, y: isize) -> bool {
        x >= 0 && x < self.width as isize && y >= 0 && y < self.height as isize
    }

    #[inline(always)]
    pub fn center(&self) -> Center {
        Center {
            x: self.width / 2,
            y: self.height / 2,
        }
    }
}

#[derive(Clone)]
pub struct Grid<Cell> {
    pub raster: Raster,
    pub cells: Vec<Cell>,
}

impl<Cell> Grid<Cell> {
    #[inline(always)]
    pub fn width(&self) -> usize {
        self.raster.width
    }
    #[inline(always)]
    pub fn height(&self) -> usize {
        self.raster.height
    }
    #[inline(always)]
    pub fn for_each(&self, f: impl FnMut(usize, usize, usize)) {
        self.raster.for_each(f);
    }
    #[inline(always)]
    pub fn index(&self, x: usize, y: usize) -> usize {
        self.raster.index(x, y)
    }
    #[inline(always)]
    pub fn center(&self) -> Center {
        self.raster.center()
    }

    #[inline(always)]
    pub fn for_each_cell(&self, mut f: impl FnMut(usize, usize, usize, &Cell)) {
        for y in 0..self.height() {
            for x in 0..self.width() {
                let index = self.index(x, y);
                let cell = self.get(x, y);
                f(x, y, index, cell);
            }
        }
    }
    #[inline(always)]
    pub fn for_each_cell_mut(&mut self, mut f: impl FnMut(usize, usize, usize, &mut Cell)) {
        for y in 0..self.height() {
            for x in 0..self.width() {
                let index = self.index(x, y);
                let cell = self.get_mut(x, y);
                f(x, y, index, cell);
            }
        }
    }

    #[inline(always)]
    pub fn for_each_in_kernel(
        &self,
        kernel: &Kernel,
        x: usize,
        y: usize,
        mut f: impl FnMut(usize, usize, usize, &Cell, &f32),
    ) {
        let kc = kernel.center();
        for ky in 0..kernel.height() {
            for kx in 0..kernel.width() {
                let dx = x as isize + kx as isize - kc.x as isize;
                let dy = y as isize + ky as isize - kc.y as isize;
                if !self.raster.is_inside(dx, dy) {
                    continue;
                }
                let index = self.index(dx as usize, dy as usize);
                let cell = self.get(dx as usize, dy as usize);
                let weight = kernel.get(kx, ky);
                f(kx, ky, index, cell, weight);
            }
        }
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
}

macro_rules! convolve_kernel {
    ($self:expr, $kernel:expr, $x:expr, $y:expr, $new_cell:expr) => {{
        let kc = $kernel.center();
        for ky in 0..$kernel.height() {
            for kx in 0..$kernel.width() {
                let dx = $x as isize + kx as isize - kc.x as isize;
                let dy = $y as isize + ky as isize - kc.y as isize;
                if !$self.raster.is_inside(dx, dy) {
                    continue;
                }
                let neighbour = $self.get(dx as usize, dy as usize);
                let weight = $kernel.cells[ky * $kernel.width() + kx];
                $new_cell.r += neighbour.r * weight;
                $new_cell.g += neighbour.g * weight;
                $new_cell.b += neighbour.b * weight;
                $new_cell.a += neighbour.a * weight;
            }
        }
    }};
}

impl Grid<RGBA> {
    pub fn convolve_fft(&mut self, kernel: &Kernel) {
        let (grid_width, grid_height) = (self.width(), self.height());
        let (kernel_width, kernel_height) = (kernel.width(), kernel.height());
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

        let padded_raster = Raster {
            width: padded_width,
            height: padded_height,
        };

        let time = Instant::now();
        let mut grid_r = vec![Complex::zero(); padded_width * padded_height];
        let mut grid_g = vec![Complex::zero(); padded_width * padded_height];
        let mut grid_b = vec![Complex::zero(); padded_width * padded_height];
        let mut grid_a = vec![Complex::zero(); padded_width * padded_height];
        let mut kernel_complex = vec![Complex::zero(); padded_width * padded_height];

        self.for_each_cell(|x, y, _, color| {
            let padded_index = y * padded_width + x;
            grid_r[padded_index] = Complex::new(color.r, 0.0);
            grid_g[padded_index] = Complex::new(color.g, 0.0);
            grid_b[padded_index] = Complex::new(color.b, 0.0);
            grid_a[padded_index] = Complex::new(color.a, 0.0);
        });
        kernel.for_each_cell(|x, y, _, cell| {
            let padded_index = y * padded_width + x;
            kernel_complex[padded_index] = Complex::new(*cell, 0.0);
        });
        println!("Preparation took {:?}", time.elapsed());
        let time = Instant::now();
        fft.process(&mut grid_r);
        fft.process(&mut grid_g);
        fft.process(&mut grid_b);
        fft.process(&mut grid_a);
        fft.process(&mut kernel_complex);
        println!("FFT processing took {:?}", time.elapsed());

        padded_raster.for_each(|_, _, i| {
            grid_r[i] *= kernel_complex[i];
            grid_g[i] *= kernel_complex[i];
            grid_b[i] *= kernel_complex[i];
            grid_a[i] *= kernel_complex[i];
        });

        let time = Instant::now();
        ifft.process(&mut grid_r);
        ifft.process(&mut grid_g);
        ifft.process(&mut grid_b);
        ifft.process(&mut grid_a);
        println!("IFFT processing took {:?}", time.elapsed());

        let kc = kernel.center();
        self.for_each_cell_mut(|x, y, _, cell| {
            let dx = x as i32 + kc.x as i32;
            let dy = y as i32 + kc.y as i32;
            let index = dy as usize * padded_width + dx as usize;
            let r_value = grid_r[index].re / (padded_width * padded_height) as f32;
            let g_value = grid_g[index].re / (padded_width * padded_height) as f32;
            let b_value = grid_b[index].re / (padded_width * padded_height) as f32;
            let a_value = grid_a[index].re / (padded_width * padded_height) as f32;

            cell.r = r_value.clamp(0.0, 1.0);
            cell.g = g_value.clamp(0.0, 1.0);
            cell.b = b_value.clamp(0.0, 1.0);
            cell.a = a_value.clamp(0.0, 1.0);
        });
    }

    pub fn convolve_par(&mut self, kernel: &Kernel) {
        let width = self.width();
        let height = self.height();
        let mut new_cells = vec![RGBA::ZERO; width * height];

        new_cells
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let mut new_cell = RGBA::ZERO;

                    convolve_kernel!(self, kernel, x, y, new_cell);

                    row[x] = RGBA {
                        r: new_cell.r.clamp(0.0, 1.0),
                        g: new_cell.g.clamp(0.0, 1.0),
                        b: new_cell.b.clamp(0.0, 1.0),
                        a: new_cell.a.clamp(0.0, 1.0),
                    };
                }
            });

        self.cells = new_cells;
    }

    pub fn convolve(&mut self, kernel: &Kernel) {
        let width = self.width();
        let height = self.height();
        let mut new_cells = vec![RGBA::ZERO; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut new_cell = RGBA::ZERO;
                convolve_kernel!(self, kernel, x, y, new_cell);
                let index = self.index(x, y);
                new_cells[index] = new_cell;
            }
        }
        self.cells = new_cells;
    }

    pub fn new_random(width: usize, height: usize) -> Grid<RGBA> {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut grid = Grid {
            raster: Raster { width, height },
            cells: vec![RGBA::ZERO; width * height],
        };

        grid.for_each_cell_mut(|_, _, _, cell| {
            cell.r = rng.gen_range(0..255) as f32 / 255.0;
            cell.g = rng.gen_range(0..255) as f32 / 255.0;
            cell.b = rng.gen_range(0..255) as f32 / 255.0;
            cell.a = 1.0;
        });

        grid
    }
}
