//! Embedding Layer.
use candle::{Result, Tensor};

#[derive(Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = Tensor::embedding(&indexes, &self.embeddings)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}
