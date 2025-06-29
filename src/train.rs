use std::fs::remove_dir_all;

use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

use crate::{
    data::{Urban8kBatch, Urban8kBatcher, Urban8kDataset},
    model::{Model, ModelConfig},
};

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<Urban8kBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Urban8kBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.stfts, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Urban8kBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: Urban8kBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.stfts, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    train_dir: &str,
    test_dir: &str,
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    remove_dir_all(artifact_dir).ok();
    config.save(artifact_dir).unwrap();

    B::seed(config.seed);

    let batcher = Urban8kBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Urban8kDataset::from_dir(train_dir));

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Urban8kDataset::from_dir(test_dir));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
