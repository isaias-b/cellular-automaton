use crate::grid::*;
use crate::kernel::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};
use std::{fmt::Debug, time::Instant};

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

impl Grid<RGBA> {
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

impl Convolver<RGBA> for SimpleConvolver {
    #[inline(always)]
    fn convolve(&self, grid: &mut Grid<RGBA>, kernel: &Kernel) {
        let width = grid.width();
        let height = grid.height();
        let mut new_cells = vec![RGBA::ZERO; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut new_cell = RGBA::ZERO;
                convolve_kernel!(grid, kernel, x, y, new_cell);
                let index = grid.index(x, y);
                new_cells[index] = new_cell;
            }
        }
        grid.cells = new_cells;
    }
}

impl Convolver<RGBA> for ParConvolver {
    #[inline(always)]
    fn convolve(&self, grid: &mut Grid<RGBA>, kernel: &Kernel) {
        let width = grid.width();
        let height = grid.height();
        let mut new_cells = vec![RGBA::ZERO; width * height];

        new_cells
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..width {
                    let mut new_cell = RGBA::ZERO;

                    convolve_kernel!(grid, kernel, x, y, new_cell);

                    row[x] = RGBA {
                        r: new_cell.r.clamp(0.0, 1.0),
                        g: new_cell.g.clamp(0.0, 1.0),
                        b: new_cell.b.clamp(0.0, 1.0),
                        a: new_cell.a.clamp(0.0, 1.0),
                    };
                }
            });

        grid.cells = new_cells;
    }
}

impl Convolver<RGBA> for FftConvolver {
    #[inline(always)]
    fn convolve(&self, grid: &mut Grid<RGBA>, kernel: &Kernel) {
        let (grid_width, grid_height) = (grid.width(), grid.height());
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

        grid.for_each_cell(|x, y, _, color| {
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
        grid.for_each_cell_mut(|x, y, _, cell| {
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
}
