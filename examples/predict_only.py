# MIT License
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Utility script for generating model predictions only.

This example demonstrates how to run :mod:`lighteval` to produce model
predictions without computing the associated metrics.  The predictions are
stored using :class:`~lighteval.logging.evaluation_tracker.EvaluationTracker`
and can later be reused to compute metrics with the standard evaluation
pipeline by passing ``--load-responses-from-details-date-id`` to ``lighteval``.

Usage example::

    python predict_only.py examples/model_configs/transformers_model.yaml \
        test|gsm8k --output-dir predictions

The resulting files in ``predictions`` can then be consumed by the regular
``lighteval`` CLI by setting ``--load-responses-from-details-date-id`` to the
folder timestamp found in ``predictions/details/<model_name>/``.
"""

from __future__ import annotations

import argparse

import yaml

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.models.utils import ModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions with lighteval without computing metrics.",
    )
    parser.add_argument(
        "model_args",
        type=str,
        help=(
            "Model arguments in the form key=value,... or a path to a YAML config "
            "file (see examples/model_configs/transformers_model.yaml)."
        ),
    )
    parser.add_argument(
        "tasks",
        type=str,
        help="Comma separated list of tasks or path to a task list file.",
    )
    parser.add_argument(
        "--custom-tasks",
        type=str,
        default=None,
        help="Path to directory containing custom task definitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where prediction details will be stored.",
    )
    parser.add_argument(
        "--dataset-loading-processes",
        type=int,
        default=1,
        help="Number of processes to use for dataset loading.",
    )
    parser.add_argument(
        "--num-fewshot-seeds",
        type=int,
        default=1,
        help="Number of seeds to use for few-shot evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples per task (useful for debugging).",
    )
    return parser.parse_args()


def build_model_config(model_args: str) -> TransformersModelConfig:
    """Construct a :class:`TransformersModelConfig` from CLI arguments."""
    if model_args.endswith(".yaml"):
        with open(model_args, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)["model_parameters"]
    else:
        config_dict = ModelConfig._parse_args(model_args)
    return TransformersModelConfig(**config_dict)


def main() -> None:
    args = parse_args()

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=True,
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        dataset_loading_processes=args.dataset_loading_processes,
        custom_tasks_directory=args.custom_tasks,
        num_fewshot_seeds=args.num_fewshot_seeds,
        max_samples=args.max_samples,
    )

    model_config = build_model_config(args.model_args)

    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    evaluation_tracker.general_config_logger.log_args_info(
        num_fewshot_seeds=args.num_fewshot_seeds,
        max_samples=args.max_samples,
        job_id="0",
        config=model_config,
    )

    outputs = pipeline._run_model()

    for sampling_method, responses in outputs.items():
        docs = pipeline.sampling_docs[sampling_method]
        for doc, response in zip(docs, responses):
            evaluation_tracker.details_logger.log(doc.task_name, doc, response, {})

    evaluation_tracker.general_config_logger.log_end_time()
    evaluation_tracker.details_logger.aggregate()
    evaluation_tracker.save()


if __name__ == "__main__":  # pragma: no cover - example script
    main()
