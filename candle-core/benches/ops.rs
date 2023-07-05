#![feature(test)]

extern crate test;
use candle::{DType, Device, Tensor};
use test::{black_box, Bencher};

#[bench]
fn bench_gelu_cpu_f32(b: &mut Bencher) {
    let mut tensor = Tensor::zeros(vec![110, 768], DType::F32, &Device::Cpu).unwrap();
    b.iter(|| {
        black_box(tensor.gelu().unwrap());
    });
}
