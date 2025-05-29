# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval: Cybersecurity Multiple Choice Questions

This file defines tasks for evaluating models on cybersecurity-related multiple-choice questions
based on various sources like cybersecurity roadmaps, Wikipedia, MITRE ATT&CK, etc.
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Define the English letter indices for choices
ENGLISH_LETTER_INDICES = ["A", "B", "C", "D"]

# Define the subsets for the cybersecurity evaluation
# These correspond to the 'config_name' in your dataset card
CYBERSEC_SUBSETS = [
    "cybersecurity_roadmap",
    "cybersecurity_wikipedia",
    "mitre_attck",
    "mitre_capec",
    "mitre_cwe",
    "owasp",
]


def cybersec_prompt_fn(line: dict, task_name: str = None, include_context: bool = True) -> Doc:
    """
    Processes a line from the cybersecurity dataset to create a Doc object for MMLU-style evaluation.

    Args:
        line: A dictionary representing a sample from the dataset.
              Expected keys: "content", "question", "answers" (dict with "A", "B", "C", "D"), "solution" (str "A"-"D").
        task_name: The name of the task.
        include_context: Whether to include the 'content' field in the prompt.

    Returns:
        A Doc object containing the formatted query, choices (letters), and gold standard index.
    """
    content = line.get("content", "")
    question = line["question"]
    answers_dict = line["answers"]  # e.g. {"A": "text A", "B": "text B", ...}
    solution_letter = line["solution"]  # e.g. "A"

    query_parts = []
    if include_context and content:
        query_parts.append("Context: " + content)
    query_parts.append("Question: " + question)

    choices_str_parts = []
    for letter in ENGLISH_LETTER_INDICES:
        choice_text = answers_dict.get(letter)
        if choice_text is None:
            # This case should ideally not happen based on the dataset card structure
            raise ValueError(f"Missing answer for choice {letter} in line: {line}")
        choices_str_parts.append(f"{letter}. {choice_text}")

    instructions = "Answer with the option letter from the given choices directly."

    # Construct the MMLU-style query
    # Example:
    # Instructions
    # Context (if any)
    #
    # Question text
    # A. Answer text for A
    # B. Answer text for B
    # C. Answer text for C
    # D. Answer text for D
    #
    # Answer:
    full_query = instructions + "\n" + "\n\n".join(query_parts) + "\n" + "\n".join(choices_str_parts) + "\n\nAnswer:"

    # Choices for the Doc object are the letters themselves, with a leading space.
    # This is a common format for MMLU-style evaluations where the model predicts the letter.
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        # solution_letter is "A", "B", etc. We need its index in ENGLISH_LETTER_INDICES
        gold_index = ENGLISH_LETTER_INDICES.index(solution_letter)
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution_letter}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=full_query,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"id": line["id"]},
        instruction=instructions,
    )


class CustomCybersecEvalTask(LightevalTaskConfig):
    """
    Configuration for a single cybersecurity evaluation task subset.
    """

    def __init__(
        self,
        name: str,
        hf_subset: str,
        include_context: bool,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=lambda line, task_name: cybersec_prompt_fn(line, task_name, include_context=include_context),
            # IMPORTANT: Replace with your actual Hugging Face Hub dataset repository ID
            # For example: "my_organization/my_cybersecurity_dataset"
            hf_repo="naufalso/cybersecurity_benchmark_mcqa",
            metric=[Metrics.loglikelihood_acc_norm],  # Using standard accuracy for multiple-choice questions
            hf_avail_splits=["test"],  # As per your dataset card
            evaluation_splits=["test"],  # As per your dataset card
            few_shots_split=None,  # No few-shot examples specified
            few_shots_select=None,  # No few-shot selection strategy
            suite=["community"],  # Add this task to the community suite
            generation_size=-1,  # For multiple-choice (loglikelihood) evaluations
            stop_sequence=None,  # Not applicable for non-generative tasks
            trust_dataset=True,  # Default, set to True if you trust the dataset source implicitly
        )


# Create a list of task configurations for all defined subsets
CYBERSEC_TASKS = []
for subset in CYBERSEC_SUBSETS:
    # Version without context (default)
    CYBERSEC_TASKS.append(
        CustomCybersecEvalTask(
            name=f"cybersec_eval:{subset}",
            hf_subset=subset,
            include_context=False,
        )
    )
    # Version with context
    CYBERSEC_TASKS.append(
        CustomCybersecEvalTask(
            name=f"cybersec_eval_ctx:{subset}",
            hf_subset=subset,
            include_context=True,
        )
    )

# The table of tasks to be imported by lighteval
TASKS_TABLE = CYBERSEC_TASKS
