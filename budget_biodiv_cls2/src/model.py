"""Transformer chunk encoder + 직접 구현한 문서 Attention Pooling 모델."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class AttentionPooling(nn.Module):
    """가변 개수 chunk embedding을 학습 가능한 가중합으로 통합한다."""

    def __init__(self, hidden_size: int, attention_size: int = 256) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )

    def forward(
        self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            chunk_embeddings: ``[batch, chunks, hidden]``
            chunk_mask: 실제 chunk 위치가 True인 ``[batch, chunks]``
        """
        if not torch.all(chunk_mask.any(dim=1)):
            raise ValueError("각 문서에는 최소 한 개의 유효 chunk가 필요합니다")
        scores = self.scorer(chunk_embeddings).squeeze(-1)
        scores = scores.masked_fill(~chunk_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~chunk_mask, 0.0)
        document_embedding = torch.sum(chunk_embeddings * weights.unsqueeze(-1), dim=1)
        return document_embedding, weights


class DocumentAttentionClassifier(nn.Module):
    """각 chunk의 첫 token embedding을 attention으로 묶는 11-class 모델."""

    def __init__(
        self,
        model_name: str = "klue/roberta-base",
        num_labels: int = 10,
        attention_size: int = 256,
        dropout: float = 0.1,
        encoder_chunk_batch_size: int = 16,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.encoder_chunk_batch_size = encoder_chunk_batch_size
        self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        hidden_size = int(self.encoder.config.hidden_size)
        self.attention_pooling = AttentionPooling(hidden_size, attention_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _encode_valid_chunks(
        self,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """긴 문서의 모든 chunk를 한 번에 GPU에 올리지 않도록 micro-batch한다."""
        batch_size = self.encoder_chunk_batch_size
        if batch_size <= 0:
            batch_size = flat_input_ids.shape[0]
        embeddings: list[torch.Tensor] = []
        for start in range(0, flat_input_ids.shape[0], batch_size):
            outputs: Any = self.encoder(
                input_ids=flat_input_ids[start : start + batch_size],
                attention_mask=flat_attention_mask[start : start + batch_size],
                return_dict=True,
            )
            embeddings.append(outputs.last_hidden_state[:, 0, :])
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_count, chunk_count, sequence_length = input_ids.shape
        flat_ids = input_ids.reshape(-1, sequence_length)
        flat_masks = attention_mask.reshape(-1, sequence_length)
        flat_chunk_mask = chunk_mask.reshape(-1)
        valid_indices = flat_chunk_mask.nonzero(as_tuple=False).squeeze(-1)
        valid_embeddings = self._encode_valid_chunks(
            flat_ids.index_select(0, valid_indices),
            flat_masks.index_select(0, valid_indices),
        )
        hidden_size = valid_embeddings.shape[-1]
        padded_embeddings = valid_embeddings.new_zeros(batch_count * chunk_count, hidden_size)
        padded_embeddings = padded_embeddings.index_copy(0, valid_indices, valid_embeddings)
        padded_embeddings = padded_embeddings.view(batch_count, chunk_count, hidden_size)

        document_embedding, attention_weights = self.attention_pooling(
            padded_embeddings, chunk_mask
        )
        logits = self.classifier(self.dropout(document_embedding))
        return {"logits": logits, "attention_weights": attention_weights}
