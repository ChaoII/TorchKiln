from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from torch.utils.data import Sampler

__all__ = ["PaddleBatchSampler"]


class PaddleBatchSampler(Sampler):
    """Reproduces ``paddle.io.DistributedBatchSampler`` shuffling + sharding.

    With ``num_replicas == 1`` the index order is exactly Paddle's single-card
    order. With ``num_replicas > 1`` the dataset is interleaved across ranks the
    same way Paddle does in distributed training.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        drop_last=False,
        num_replicas=1,
        rank=0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _local_batches(self):
        n = len(self.dataset)
        nranks = max(1, self.num_replicas)
        num_samples = int(math.ceil(n * 1.0 / nranks))
        total_size = num_samples * nranks

        indices = np.arange(n).tolist()
        padding_size = total_size - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        if self.shuffle:
            np.random.RandomState(self.epoch & 0xFFFFFFFF).shuffle(indices)
            self.epoch += 1

        if nranks > 1:
            bs = self.batch_size
            last_batch_size = total_size % (bs * nranks)
            last_local_batch_size = last_batch_size // nranks
            subsampled = []
            for i in range(
                self.rank * bs, len(indices) - last_batch_size, bs * nranks
            ):
                subsampled.extend(indices[i : i + bs])
            tail = indices[len(indices) - last_batch_size :]
            subsampled.extend(
                tail[
                    self.rank
                    * last_local_batch_size : (self.rank + 1)
                    * last_local_batch_size
                ]
            )
            indices = subsampled

        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            batches.append(batch)
        return batches

    def __iter__(self):
        for batch in self._local_batches():
            yield batch

    def __len__(self):
        n = len(self.dataset)
        nranks = max(1, self.num_replicas)
        num_samples = int(math.ceil(n * 1.0 / nranks))
        if self.drop_last:
            return num_samples // self.batch_size
        return (num_samples + self.batch_size - 1) // self.batch_size
