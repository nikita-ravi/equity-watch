"""Embedding model wrapper. The model name comes from config and is swappable.

The same wrapper also exposes the model's tokenizer, so chunking is measured in
the exact tokens the model will encode.
"""

import logging

import config
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

log = logging.getLogger(__name__)

# Chunking deliberately tokenizes text far longer than the model's max length;
# HF's "sequence longer than max" notice is noise here.
hf_logging.set_verbosity_error()


class Embedder:
    def __init__(self, model_name, batch_size=64):
        log.info("loading embedding model %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=config.TORCH_DEVICE)
        self.tokenizer = self.model.tokenizer
        self.batch_size = batch_size

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self):
        return self.model.max_seq_length

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False,
                                         truncation=False))

    def encode_tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False,
                                     truncation=False)

    def decode_tokens(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def embed(self, texts):
        """Embed a list of chunk texts -> list of vectors (normalised for cosine)."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]
