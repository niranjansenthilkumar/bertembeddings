# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer
import loader

logger = logging.getLogger(__name__)


class InputExample(object):
	"""A single training/test example for multiple choice"""

	def __init__(self, example_id, question, contexts, endings, label=None):
		"""Constructs a InputExample.
		Args:
			example_id: Unique id for the example.
			contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
			question: string. The untokenized text of the second sequence (question).
			endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.example_id = example_id
		self.question = question
		self.contexts = contexts
		self.endings = endings
		self.label = label

class InputFeatures(object):
	def __init__(self, example_id, choices_features, label):
		self.example_id = example_id
		self.choices_features = [
			{
				'input_ids': input_ids,
				'input_mask': input_mask,
				'segment_ids': segment_ids
			}
			for input_ids, input_mask, segment_ids in choices_features
		]
		self.label = label



class ROCProcessor(object):
	"""Processor for the SWAG data set."""

	def get_train_examples(self, data_dir):
		"""See base class."""
		train_data = loader.load_data('data/train.csv', False)
		return self._create_examples(train_data, False)
		

	def get_dev_examples(self, data_dir):
		"""See base class."""
		valid_data = loader.load_data('data/dev.csv', False)
		return self._create_examples(valid_data, False)

	def get_test_examples(self, data_dir):
		"""See base class."""
		test_data = loader.load_data('data/test.csv', True)
		return self._create_examples(test_data, True)
		

	def get_labels(self):
		"""See base class."""
		return ["1", "2"]

	def _create_examples(self, data, test):
		"""Creates examples for the training and dev sets."""
		examples = []
		if not(test):
			for elt in data:
				example = InputExample(
						example_id=elt[0],
						question=elt[1][0] + elt[1][1] + elt[1][2],  # in the swag dataset, the
						# common beginning of each
						# choice is stored in "sent2".
						contexts = [elt[1][3], elt[1][3]],
						endings = elt[2],
						label=str(elt[3]+1)
					) # we skip the line with the column names

				examples.append(example)

		else:
			for elt in data:
				example = InputExample(
						example_id=elt[0],
						question=elt[1][0] + elt[1][1] + elt[1][2],  # in the swag dataset, the
						# common beginning of each
						# choice is stored in "sent2".
						contexts = [elt[1][3], elt[1][3]],
						endings = elt[2],
						label="1"
					) # we skip the line with the column names
				examples.append(example)

		return examples



def convert_examples_to_features(
	examples: List[InputExample],
	label_list: List[str],
	max_length: int,
	tokenizer: PreTrainedTokenizer,
	pad_token_segment_id=0,
	pad_on_left=False,
	pad_token=0,
	mask_padding_with_zero=True,
) -> List[InputFeatures]:
	"""
	Loads a data file into a list of `InputFeatures`
	"""

	label_map = {label : i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))
		choices_features = []
		for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
			text_a = context
			if example.question.find("_") != -1:
				# this is for cloze question
				text_b = example.question.replace("_", ending)
			else:
				text_b = example.question + " " + ending

			inputs = tokenizer.encode_plus(
				text_a,
				text_b,
				add_special_tokens=True,
				max_length=max_length,
			)
			if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
				logger.info('Attention! you are cropping tokens (swag task is ok). '
						'If you are training ARC and RACE and you are poping question + options,'
						'you need to try to use a bigger max seq length!')

			input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

			# Zero-pad up to the sequence length.
			padding_length = max_length - len(input_ids)
			if pad_on_left:
				input_ids = ([pad_token] * padding_length) + input_ids
				attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
				token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
			else:
				input_ids = input_ids + ([pad_token] * padding_length)
				attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
				token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

			assert len(input_ids) == max_length
			assert len(attention_mask) == max_length
			assert len(token_type_ids) == max_length
			choices_features.append((input_ids, attention_mask, token_type_ids))


		label = label_map[example.label]

		if ex_index < 2:
			logger.info("*** Example ***")
			logger.info("race_id: {}".format(example.example_id))
			for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
				logger.info("choice: {}".format(choice_idx))
				logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
				logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
				logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
				logger.info("label: {}".format(label))

		features.append(
			InputFeatures(
				example_id=example.example_id,
				choices_features=choices_features,
				label=label,
			)
		)

	return features





processors = {
	"rocprocessor": ROCProcessor,
}