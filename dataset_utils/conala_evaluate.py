# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import json
import os.path
import re

import platform
import sys, os
system = platform.system()
if system == 'Darwin':
    root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
elif system == 'Linux':
    root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
sys.path.insert(0, root_path)
from dataset_configs import ConalaLoader
from generator.generate_utils import generate_config


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
                # print(i, f"{precisions[i]:.03f}={float(matches_by_order[i]):.03f}/{possible_matches_by_order[i]}")
            else:
                precisions[i] = 0.0
    # print("========")
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    # print(bleu, precisions, bp, ratio, translation_length, reference_length)
    return (bleu, precisions, bp, ratio, translation_length, reference_length)


""" The tokenizer that we use for code submissions, from Wang Ling et al., Latent Predictor Networks for Code Generation (2016)
    @param code: string containing a code snippet
    @return: list of code tokens
"""


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def _bleu(ref, trans, subword_option=None, smooth=True, code_tokenize=False):
    assert code_tokenize
    assert not smooth
    assert len(ref) == len(trans)
    max_order = 4
    # ref_files = [ref_file]
    # reference_text = []
    # for reference_filename in ref_files:
    #     with open(reference_filename) as fh:
    #         reference_text.append(fh.readlines())
    reference_text = [ref]
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            if code_tokenize:
                reference_list.append(tokenize_for_bleu_eval(reference.strip()))
            else:
                reference_list.append(reference.strip().split())
        per_segment_references.append(reference_list)
    translations = []
    # with open(trans_file) as fh:
    #     for line in fh:
    for line in trans:
        if code_tokenize:
            translations.append(tokenize_for_bleu_eval(line.strip()))
        else:
            translations.append(line.strip().split())
    print(f'src length: {len(per_segment_references)}, tgt length: {len(translations)}')
    bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score, 2)


if __name__ == '__main__':
    args = generate_config()
    args.dataset_type = 'test'
    conala_loader = ConalaLoader()
    oracle_list = conala_loader.load_oracle_list(args.dataset_type)
    gold = [oracle['output'] for oracle in oracle_list]
    pred_list = json.load(open(args.save_file, 'r'))
    pred = [pred['output'] for pred in pred_list]

    # idx = 0
    # for gold_tmp, pred_tmp in zip(gold, pred):
    #     if gold_tmp == 'getattr(test, a_string)':
    #         print(idx)
    #     idx += 1
    #     # print(gold_tmp, pred_tmp)

    bleu_score = _bleu(gold, pred, smooth=False, code_tokenize=True)
    print('bleu score:', bleu_score)
