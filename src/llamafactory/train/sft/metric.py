# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
import re
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def compute_accuracy(eval_preds: "EvalPrediction") -> Dict[str, float]:
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    accuracies = []
    for i in range(len(preds)):
        pred, label = preds[i, :-1], labels[i, 1:]
        label_mask = label != IGNORE_INDEX
        accuracies.append(np.mean(pred[label_mask] == label[label_mask]))

    return {"accuracy": float(np.mean(accuracies))}


def eval_logit_processor(
    logits: "torch.Tensor", labels: "torch.Tensor"
) -> "torch.Tensor":
    logits = logits[0] if isinstance(logits, (list, tuple)) else logits
    return torch.argmax(logits, dim=-1)


def compute_metrics_for_label(pred_str, label_str):
    match = re.search(r"\b([ABCDE])\b", pred_str)
    if match:
        predicted_answer = match.group(1)
        return predicted_answer == label_str
    return False


COMPUTE_METHOD = {"func": compute_metrics_for_label}


@dataclass
class ComputeLabelMetrics:
    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: "EvalPrediction") -> Dict[str, float]:
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        accuracies = []
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            decoded_preds = self.tokenizer.batch_decode(
                pred[label_mask], skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                label[label_mask], skip_special_tokens=True
            )
            func = COMPUTE_METHOD["func"]
            accuracies.append(
                np.mean(func("".join(decoded_preds), "".join(decoded_labels)))
            )
        return {"accuracy": float(np.mean(accuracies))}


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: "EvalPrediction") -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if (
                len(" ".join(hypothesis).split()) == 0
                or len(" ".join(reference).split()) == 0
            ):
                result = {
                    "rouge-1": {"f": 0.0},
                    "rouge-2": {"f": 0.0},
                    "rouge-l": {"f": 0.0},
                }
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3,
            )
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}

@dataclass
class ComputeBoolMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: "EvalPrediction") -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        score_dict = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "P":0.0,"R":0.0,"F1":0.0,"acc": 0.0}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            print('pred:'+pred,'label:'+label)
            if label.lower() == "true":
                if pred.lower() == "true":
                    score_dict["TP"] += 1
                else:
                    score_dict["FN"] += 1
            else:
                if pred.lower() == "true":
                    score_dict["FP"] += 1
                else:
                    score_dict["TN"] += 1
        
        score_dict["P"] = score_dict["TP"] / (score_dict["TP"] + score_dict["FP"]) if score_dict["TP"] + score_dict["FP"] != 0 else 0
        score_dict["R"] = score_dict["TP"] / (score_dict["TP"] + score_dict["FN"]) if score_dict["TP"] + score_dict["FN"] != 0 else 0
        score_dict["F1"] = 2 * score_dict["P"] * score_dict["R"] / (score_dict["P"] + score_dict["R"]) if score_dict["P"] + score_dict["R"] != 0 else 0
        score_dict["acc"] = (score_dict["TP"] + score_dict["TN"]) / (score_dict["TP"] + score_dict["FN"] + score_dict["FP"] + score_dict["TN"]) if score_dict["TP"] + score_dict["FN"] + score_dict["FP"] + score_dict["TN"] != 0 else 0
        
        return score_dict
    
@dataclass
class ComputeEqualMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: "EvalPrediction") -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        score_dict = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "P":0.0,"R":0.0,"F1":0.0,"acc": 0.0}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            print('pred:'+pred,'label:'+label)
            if label.lower() == pred.lower():
                score_dict["TP"] += 1
            else:
                score_dict["FN"] += 1
        
        score_dict["P"] = score_dict["TP"] / (score_dict["TP"] + score_dict["FP"]) if score_dict["TP"] + score_dict["FP"] != 0 else 0
        score_dict["R"] = score_dict["TP"] / (score_dict["TP"] + score_dict["FN"]) if score_dict["TP"] + score_dict["FN"] != 0 else 0
        score_dict["F1"] = 2 * score_dict["P"] * score_dict["R"] / (score_dict["P"] + score_dict["R"]) if score_dict["P"] + score_dict["R"] != 0 else 0
        score_dict["acc"] = (score_dict["TP"] + score_dict["TN"]) / (score_dict["TP"] + score_dict["FN"] + score_dict["FP"] + score_dict["TN"]) if score_dict["TP"] + score_dict["FN"] + score_dict["FP"] + score_dict["TN"] != 0 else 0
        
        return score_dict
    
@dataclass
class ComputeRegularMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: "EvalPrediction") -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        score_dict = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "P":0.0,"R":0.0,"F1":0.0,"acc": 0.0}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            print('pred:'+pred,'label:'+label)
            if re.search(r'\b' + re.escape(label) + r'\b', pred, re.IGNORECASE):
                score_dict["TP"] += 1
            else:
                score_dict["FN"] += 1
        
        score_dict["P"] = score_dict["TP"] / (score_dict["TP"] + score_dict["FP"]) if score_dict["TP"] + score_dict["FP"] != 0 else 0
        score_dict["R"] = score_dict["TP"] / (score_dict["TP"] + score_dict["FN"]) if score_dict["TP"] + score_dict["FN"] != 0 else 0
        score_dict["F1"] = 2 * score_dict["P"] * score_dict["R"] / (score_dict["P"] + score_dict["R"]) if score_dict["P"] + score_dict["R"] != 0 else 0
        score_dict["acc"] = (score_dict["TP"] + score_dict["TN"]) / (score_dict["TP"] + score_dict["FN"] + score_dict["FP"] + score_dict["TN"]) if score_dict["TP"] + score_dict["FN"] + score_dict["FP"] + score_dict["TN"] != 0 else 0
        
        return score_dict