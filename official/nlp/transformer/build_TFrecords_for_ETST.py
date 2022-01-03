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

"""Download and preprocess WMT17 ende training and evaluation datasets."""

import os
import random
import tarfile

# pylint: disable=g-bad-import-order

from absl import app
from absl import flags
from absl import logging
import six
from six.moves import range
from six.moves import urllib
from six.moves import zip
import tensorflow.compat.v1 as tf

from official.nlp.transformer.utils import tokenizer
from official.utils.flags import core as flags_core
# pylint: enable=g-bad-import-order
from official.nlp.transformer.utils.others import from_path_import

# from dictionary import Dictionary
name = "dictionary"
path = "/home/zchen/encyclopedia-text-style-transfer/dictionary.py"
demands = ["Dictionary"]
from_path_import(name, path, globals(), demands)


# Strings to inclue in the generated files.
_PREFIX = ""
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
# evaluation datasets are tagged as "dev" for development.
_TEST_TAG = "test"

# Number of files to split train and evaluation data
_TRAIN_SHARDS = 100
_EVAL_SHARDS = _TEST_SHARDS = 1

def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.io.gfile.GFile(path) as f:
    for line in f:
      yield line.strip()

###############################################################################
# Data preprocessing
###############################################################################
def encode_and_save_files(dico, out_path, data_path, tag, total_shards):
  """Save data from files as encoded Examples in TFrecord format.

  Args:
    subtokenizer: Subtokenizer object that will be used to encode the strings.
    out_path: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.

  Returns:
    List of all files produced.
  """
  # Create a file for each shard.
  filepaths = [
      shard_filename(out_path, tag, n + 1, total_shards)
      for n in range(total_shards)
  ]

  if all_exist(filepaths):
    logging.info("Files with tag %s already exist.", tag)
    return filepaths

  logging.info("Saving files with tag %s.", tag)

  # Write examples to each shard in round robin order.
  tmp_filepaths = [six.ensure_str(fname) + ".incomplete" for fname in filepaths]
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
  counter, shard = 0, 0
  for counter, input_line in enumerate(txt_line_iterator(data_path)):
    if counter > 0 and counter % 1000000 == 0:
      logging.info("\tSaving case %d.", counter)
    example = dict_to_example({
        "inputs": [dico.index(token) for token in input_line.split()]
    })
    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % total_shards
  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  logging.info("Saved %d Examples", counter + 1)
  return filepaths


def shard_filename(path, tag, shard_num, total_shards):
  """Create filename for data shard."""
  return os.path.join(
      path, "%s-%.5d-of-%.5d" % (tag, shard_num, total_shards))


def shuffle_records(fname):
  """Shuffle records in a single file."""
  logging.info("Shuffling records in file %s", fname)

  # Rename file prior to shuffling
  tmp_fname = six.ensure_str(fname) + ".unshuffled"
  tf.gfile.Rename(fname, tmp_fname)

  reader = tf.io.tf_record_iterator(tmp_fname)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      logging.info("\tRead: %d", len(records))

  random.shuffle(records)

  # Write shuffled records to original file name
  with tf.python_io.TFRecordWriter(fname) as w:
    for count, record in enumerate(records):
      w.write(record)
      if count > 0 and count % 100000 == 0:
        logging.info("\tWriting record: %d", count)

  tf.gfile.Remove(tmp_fname)


def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def all_exist(filepaths):
  """Returns true if all files in the list exist."""
  for fname in filepaths:
    if not tf.gfile.Exists(fname):
      return False
  return True


def make_dir(path):
  if not tf.gfile.Exists(path):
    logging.info("Creating directory %s", path)
    tf.gfile.MakeDirs(path)


def main(unused_argv):
  make_dir(FLAGS.out_path)

  # Numericalize and save data as Examples in the TFRecord format.
  logging.info("Preprocessing and saving data")
  
  dico = Dictionary.read_vocab(FLAGS.voc_path)
  logging.info("")
  
  train_path, eval_path, test_path = [os.path.join(FLAGS.data_path, split) for split in ["train", "valid", "test"]]
  train_tfrecord_files = encode_and_save_files(dico, FLAGS.out_path,
                                               train_path, _TRAIN_TAG,
                                               _TRAIN_SHARDS)
  encode_and_save_files(dico, FLAGS.out_path, eval_path,
                        _EVAL_TAG, _EVAL_SHARDS)
  encode_and_save_files(dico, FLAGS.out_path, test_path,
                        _TEST_TAG, _TEST_SHARDS)

  for fname in train_tfrecord_files:
    shuffle_records(fname)


def define_build_TFrecord_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="out_path",
      default="/tmp/translate_ende",
      help=flags_core.help_wrap(
          "Directory for where the translate_ende_wmt32k dataset is saved."))
  flags.DEFINE_string(
      name="data_path",
      default="",
      help=flags_core.help_wrap(
          "Tokenized data file path."))
  flags.DEFINE_string(
      name="voc_path",
      default="",
      help=flags_core.help_wrap(
          "Vocabulary file path."))

if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_build_TFrecord_flags()
  FLAGS = flags.FLAGS
  app.run(main)
