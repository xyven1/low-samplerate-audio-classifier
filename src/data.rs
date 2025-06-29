use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor},
};

#[derive(Clone, Debug)]
pub struct Urban8kItem {
    pub stft: Vec<f32>,
    pub time_steps: usize,
    pub frequency_bins: usize,
    pub label: u8,
}

pub struct Urban8kDataset {
    pub items: Vec<Urban8kItem>,
}

impl Urban8kDataset {
    pub fn from_dir(dir: &str) -> Self {
        todo!()
    }
}

impl Dataset<Urban8kItem> for Urban8kDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<Urban8kItem> {
        self.items.get(index).cloned()
    }
}

#[derive(Clone, Default)]
pub struct Urban8kBatcher {}

#[derive(Clone, Debug)]
pub struct Urban8kBatch<B: Backend> {
    pub stfts: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, Urban8kItem, Urban8kBatch<B>> for Urban8kBatcher {
    fn batch(&self, items: Vec<Urban8kItem>, device: &B::Device) -> Urban8kBatch<B> {
        let stfts = items
            .iter()
            .map(|item| {
                Tensor::<B, 2>::from_data(item.stft.as_slice(), device).reshape([
                    1,
                    item.time_steps,
                    item.frequency_bins,
                ])
            })
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let stfts = Tensor::cat(stfts, 0);
        let targets = Tensor::cat(targets, 0);

        Urban8kBatch { stfts, targets }
    }
}
