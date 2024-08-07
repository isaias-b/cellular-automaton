use crate::Grid;
use crate::Raster;

pub type Kernel = Grid<f32>;

impl Kernel {
    pub fn gauss3() -> Kernel {
        #[rustfmt::skip]
        let cells = vec![
            1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
            2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
            1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
        ];
        Kernel {
            raster: Raster {
                width: 3,
                height: 3,
            },
            cells,
        }
    }
    pub fn gauss5() -> Kernel {
        #[rustfmt::skip]
        let cells = vec![
            1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
            4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
            6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
            4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
            1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
        ];
        Kernel {
            raster: Raster {
                width: 5,
                height: 5,
            },
            cells,
        }
    }
    pub fn gauss7() -> Kernel {
        #[rustfmt::skip]
        let cells = vec![
            1.0 / 4096.0, 6.0 / 4096.0, 15.0 / 4096.0, 20.0 / 4096.0, 15.0 / 4096.0, 6.0 / 4096.0, 1.0 / 4096.0,
            6.0 / 4096.0, 36.0 / 4096.0, 90.0 / 4096.0, 120.0 / 4096.0, 90.0 / 4096.0, 36.0 / 4096.0, 6.0 / 4096.0,
            15.0 / 4096.0, 90.0 / 4096.0, 225.0 / 4096.0, 300.0 / 4096.0, 225.0 / 4096.0, 90.0 / 4096.0, 15.0 / 4096.0,
            20.0 / 4096.0, 120.0 / 4096.0, 300.0 / 4096.0, 400.0 / 4096.0, 300.0 / 4096.0, 120.0 / 4096.0, 20.0 / 4096.0,
            15.0 / 4096.0, 90.0 / 4096.0, 225.0 / 4096.0, 300.0 / 4096.0, 225.0 / 4096.0, 90.0 / 4096.0, 15.0 / 4096.0,
            6.0 / 4096.0, 36.0 / 4096.0, 90.0 / 4096.0, 120.0 / 4096.0, 90.0 / 4096.0, 36.0 / 4096.0, 6.0 / 4096.0,
            1.0 / 4096.0, 6.0 / 4096.0, 15.0 / 4096.0, 20.0 / 4096.0, 15.0 / 4096.0, 6.0 / 4096.0, 1.0 / 4096.0,
        ];
        Kernel {
            raster: Raster {
                width: 7,
                height: 7,
            },
            cells,
        }
    }
}
