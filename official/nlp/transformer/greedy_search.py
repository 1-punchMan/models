# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Beam search to find the translated sequence with the highest probability."""

import numpy as np
import tensorflow as tf

  def greedy_search(
    symbols_to_logits_fn,
    initial_ids,
    initial_soft_seq,
    initial_cache,
    vocab_size,
    max_decode_length,
    eos_id,
    dtype=tf.float32
    ):
    batch_size = tf.shape(initial_ids)[0]
    step = 0
    seq = tf.expand_dims(initial_ids, -1)  # (batch_size, cur_len)
    soft_seq = tf.broadcast_to(initial_soft_seq, [batch_size, 1, vocab_size])  # (batch_size, cur_len, vocab_size)
    cache = initial_cache
    while step < max_decode_length and not tf.math.reduce_all(finished):
      # Get logits for the next candidate IDs for the sequences. Get the
      # new cache values at the same time.
      logits, cache = self.symbols_to_logits_fn(seq, step, cache)

      best_ids = tf.math.argmax(logits, axis=-1)  # (batch_size, 1)

      # Convert logits to normalized probs
      # Use softmax to approximate one-hot
      b = 10
      logits = tf.exp(b * logits)
      probs = logits / tf.reduce_sum(logits, axis=-1, keepdims=True)  # (batch_size, 1, vocab_size)

      # Append the most probable IDs to the topk sequences
      seq = tf.concat([seq, best_ids], axis=1)
      soft_seq = tf.concat([soft_seq, probs], axis=1)
      
      finished |= tf.equal(best_ids, self.eos_id)
      step += 1

    return seq, soft_seq