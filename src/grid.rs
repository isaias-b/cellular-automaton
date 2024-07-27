use crate::kernel::Kernel;
use std::{fmt::Debug, time::Instant};

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

pub trait Convolver<Cell> {
    fn convolve(&self, grid: &mut Grid<Cell>, kernel: &Kernel);
}
#[derive(Debug)]
pub struct SimpleConvolver;

#[derive(Debug)]
pub struct FftConvolver;

#[derive(Debug)]
pub struct ParConvolver;

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

    #[inline(always)]
    pub fn convolve(&mut self, kernel: &Kernel, convolver: impl Convolver<Cell> + Debug) {
        let time = Instant::now();
        convolver.convolve(self, kernel);
        println!(
            "Convolution with {:?} took {:?} for grid {}x{} kernel {}x{}",
            convolver,
            time.elapsed(),
            self.width(),
            self.height(),
            kernel.width(),
            kernel.height(),
        );
    }
}
