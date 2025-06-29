use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    prelude::*,
    tensor::Distribution,
};
use low_samplerate_audio_classifier::model::ModelConfig;

fn main() {
    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::DiscreteGpu(0);

    let model = ModelConfig {
        num_classes: 10,
        hidden_size: 128,
        dropout: 0.5,
    }
    .init::<Backend>(&device);

    let res = model.forward_classification(
        Tensor::random([32, 28, 28], Distribution::Uniform(-1.0, 1.0), &device),
        Tensor::zeros([32], &device),
    );
    println!("{}", res.output);
}
