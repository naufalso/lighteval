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
"""Utility script for computing metrics from saved prediction details.

This example complements :mod:`predict_only.py` by showing how to load the
prediction details generated previously and compute the evaluation metrics
without running model inference again.

Usage example::

    python evaluate_from_details.py test|gsm8k \
        --details-date-id 2024-04-20T12-00-00 \
        --model-name my-model \
        --output-dir predictions

The ``details-date-id`` corresponds to the timestamp folder produced by
``predict_only.py`` under ``<output-dir>/details/<model-name>/``.
"""

from __future__ import annotations

import argparse

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.dummy.dummy_model import DummyModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics from stored prediction details.",
    )
    parser.add_argument(
        "tasks",
        type=str,
        help="Comma separated list of tasks or path to a task list file.",
    )
    parser.add_argument(
        "--details-date-id",
        type=str,
        required=True,
        help="Timestamp folder created by predict_only.py.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Model name used when generating predictions. It must match the folder name inside <output-dir>/details/",
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory containing the prediction details.",
    )
    parser.add_argument(
        "--custom-tasks",
        type=str,
        default=None,
        help="Path to directory containing custom task definitions.",
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


def main() -> None:
    args = parse_args()

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir,
        save_details=False,  # Details already exist from predict_only.py
        push_to_hub=False,
        push_to_tensorboard=False,
        public=False,
        hub_results_org=None,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        dataset_loading_processes=args.dataset_loading_processes,
        custom_tasks_directory=args.custom_tasks,
        num_fewshot_seeds=args.num_fewshot_seeds,
        max_samples=args.max_samples,
        load_responses_from_details_date_id=args.details_date_id,
    )

    dummy_config = DummyModelConfig()

    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=dummy_config,
    )

    # Ensure tracker looks for details under the original model name
    evaluation_tracker.general_config_logger.model_name = args.model_name

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":  # pragma: no cover - example script
    main()
