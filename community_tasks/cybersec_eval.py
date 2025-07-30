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

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants
ENGLISH_LETTER_INDICES = ["A", "B", "C", "D"]

# SecEval supports multi-answer choices
SECEVAL_ENGLISH_LETTER_INDICES = [
    "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", 
    "ABC", "ABD", "ACD", "BCD", "ABCD", ""
]

# Define the subsets for the cybersecurity evaluation
CYBERSEC_SUBSETS = [
    "cybersecurity_roadmap",
    "cybersecurity_wikipedia",
    "mitre_attck",
    "mitre_capec",
    "mitre_cwe",
    "owasp",
    "kali_docs",
    "kali_tools",
    "tldr_en",
    "man_pages",
    "practical_pentesting_techniques",
    "nullbyte",
    "hackingarticles"
]

CTIBENCH_SUBSETS = [
    "cti-mcq",
    "cti-rcm",
    "cti-vsp",
    "cti-ate",
]

SECURE_SUBSETS = [
    "CWET",
    "MAET",
]

# SecEval few-shot examples
SECEVAL_FEW_SHOT_EXAMPLES = """
Question: You are reviewing a piece of code to identify potential vulnerabilities related to NULL pointer dereferences. Which of the following strategies should you consider implementing to mitigate the risk? Choose all that apply. A: Use a programming language that inherently manages pointers and prevents direct memory access. B: Apply input validation to all variables and data stores that receive data from external sources to ensure they are initialized to expected values. C: Explicitly initialize all variables and other data stores, either during declaration or just before the first usage. D: Implement a custom memory management system that automatically assigns a unique identifier to each pointer.
Answer: ABC 
Question: Analyze the code snippet provided. Which of the following practices should be implemented in a multithreaded application to prevent NULL pointer dereference vulnerabilities? Select all that apply.A: Ensure that proper locking APIs are used to lock the pointer before the if statement and unlock when it has finished to prevent race conditions. B: Check the return value of all functions that return a value, including network-related functions, and verify that it is non-null before using it.C: Use automated static analysis tools that target this type of weakness, understanding that while not perfect, they can still be effective.D: Verify that a non-nil 'response' is present before deferring 'response.Body.Close()' to handle cases where the 'Do' method returns an error.
Answer: ABCD 
Question: In a bank management application, a method called `createBankAccount` allows the creation of bank accounts without user identity verification or authority checks. Based on common security practices, which of the following measures would improve security?A: Implement a centralized authentication mechanism for operations that require a proven user identity.B: Duplicate client-side authentication checks on the server side to prevent bypassing of security checks.C: Use a vetted library or framework that offers robust authentication capabilities to prevent security weaknesses.D: Apply strong authentication requirements for all users accessing data stored in the cloud.
Answer: ABCD
Question: What are the potential consequences if an application handling sensitive data does not require authentication for data access, and what are the appropriate mitigation strategies?A: Attackers could easily access sensitive data from any location; require strong authentication for data access using cloud provider controls.B: Security checks might be bypassed if only performed on the client side; implement checks on both client and server sides.C: Unauthenticated users could alter product functionality; do not use authentication for critical functionality in products.D: Sensitive data may be accessed without proper credentials; utilize authentication capabilities provided by the framework or operating system.
Answer: ABD
Question: To prevent security vulnerabilities related to deserialization of untrusted data in a Java application, which of the following practices should a developer implement?A: Use the signing/sealing features of the programming language to assure that deserialized data has not been tainted.B: Explicitly define a final readObject() method to throw an exception and prevent deserialization.C: Populate a new object by deserializing data to ensure data flows through safe input validation functions.D: Make fields transient to protect them from deserialization and prevent carrying over sensitive variables.
Answer: ABCD
""".strip()


# ============ Utility Functions ============

def validate_mcq_line(line: Dict, required_keys: List[str]) -> None:
    """Validate that a line contains all required keys."""
    missing_keys = [key for key in required_keys if key not in line]
    if missing_keys:
        raise ValueError(f"Missing required keys in dataset line: {missing_keys}")


def _extract_rcm(text: str) -> Tuple[str, bool]:
    """Extract CWE ID from text."""
    cwe_pattern = r'CWE-\d+'
    matches = re.findall(cwe_pattern, text)
    if matches:
        return matches[-1], True
    return text, False


def _extract_vsp(text: str) -> Tuple[str, bool]:
    """Extract CVSS v3.1 vector string from text."""
    cvss_pattern = r'AV:[A-Za-z]+/AC:[A-Za-z]+/PR:[A-Za-z]+/UI:[A-Za-z]+/S:[A-Za-z]+/C:[A-Za-z]+/I:[A-Za-z]+/A:[A-Za-z]+'
    matches = re.findall(cvss_pattern, text)
    if matches:
        return matches[-1], True
    return text, False


def _extract_mitre_techniques(text: str) -> Tuple[List[str], bool]:
    """
    Extract MITRE ATT&CK technique IDs from text.
    
    Args:
        text: Input text containing potential MITRE technique IDs
        
    Returns:
        Tuple of (list of technique IDs, boolean indicating if any found)
    """
    # Pattern to match MITRE technique IDs (T followed by 4 digits, optionally with .xxx subtechnique)
    technique_pattern = r'T\d{4}(?:\.\d{3})?'
    
    # Find all matches in the text
    matches = re.findall(technique_pattern, text)
    
    # Remove duplicates while preserving order
    unique_techniques = []
    seen = set()
    for technique in matches:
        # Convert subtechniques to main techniques (T1234.001 -> T1234)
        main_technique = technique.split('.')[0]
        if main_technique not in seen:
            unique_techniques.append(main_technique)
            seen.add(main_technique)
    
    return unique_techniques, len(unique_techniques) > 0


def _parse_technique_list(technique_string: str) -> List[str]:
    """
    Parse a comma-separated string of technique IDs into a list.
    
    Args:
        technique_string: String like "T1437, T1624, T1643"
        
    Returns:
        List of technique IDs
    """
    if not technique_string.strip():
        return []
    
    # Split by comma and clean up whitespace
    techniques = [t.strip() for t in technique_string.split(',')]
    
    # Filter out empty strings and ensure valid format
    valid_techniques = []
    for technique in techniques:
        if re.match(r'T\d{4}', technique):
            # Convert to main technique if it's a subtechnique
            main_technique = technique.split('.')[0]
            valid_techniques.append(main_technique)
    
    return valid_techniques


# ============ Prompt Functions ============

def cybersec_prompt_fn(line: Dict, task_name: Optional[str] = None, include_context: bool = True) -> Doc:
    """
    Processes a line from the cybersecurity dataset to create a Doc object for MMLU-style evaluation.

    Args:
        line: A dictionary representing a sample from the dataset.
              Expected keys: "content", "question", "answers" (dict with "A", "B", "C", "D"), "solution" (str "A"-"D").
        task_name: The name of the task.
        include_context: Whether to include the 'content' field in the prompt.

    Returns:
        A Doc object containing the formatted query, choices (letters), and gold standard index.
        
    Raises:
        ValueError: If required keys are missing or invalid solution letter is provided.
    """
    validate_mcq_line(line, ["question", "answers", "solution"])
    
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
            raise ValueError(f"Missing answer for choice {letter} in line: {line}")
        choices_str_parts.append(f"{letter}. {choice_text}")

    instructions = "You are given multiple choice questions. Answer with the option letter from the given choices directly."

    # Construct the MMLU-style query
    full_query = instructions + "\n" + "\n\n".join(query_parts) + "\n" + "\n".join(choices_str_parts) + "\nAnswer:"

    # Choices for the Doc object are the letters themselves, with a leading space.
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
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
        specific={"id": line.get("id")},
        instruction=instructions,
    )

def secure_mcq_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """
    Processes a line from the Secure MCQ dataset to create a Doc object for MMLU-style evaluation.

    Args:
        line: A dictionary representing a sample from the dataset.
              Expected keys: "question", "answers" (dict with "A", "B", "C", "D"), "solution" (str "A"-"D").
        task_name: The name of the task.

    Returns:
        A Doc object containing the formatted query, choices (letters), and gold standard index.
        
    Raises:
        ValueError: If required keys are missing or invalid solution letter is provided.
    """
    validate_mcq_line(line, ["question", "options", "answer"])
    
    question = line["question"]
    answers_list = line["options"]  # e.g. ["Text A", "Text B", "Text C", "Text D"]
    solution_letter = line["answer"]  # e.g. "A"

    query_parts = [f"Question: {question}"]

    choices_str_parts = []
    for letter, choice_text in zip(ENGLISH_LETTER_INDICES, answers_list):
        choices_str_parts.append(f"{letter}. {choice_text}")

    instructions = "You are given multiple choice questions. Answer with the option letter from the given choices directly."

    # Construct the MMLU-style query
    full_query = instructions + "\n" + "\n".join(query_parts) + "\n" + "\n".join(choices_str_parts) + "\nAnswer:"

    # Choices for the Doc object are the letters themselves, with a leading space.
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
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
        specific={"id": line.get("id")},
        instruction=instructions,
    )


# ============ Task Configuration Classes ============

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
            hf_repo="naufalso/cybersecurity_benchmark_mcqa_cleaned",
            metrics=[Metrics.loglikelihood_acc_norm],  # Using standard accuracy for multiple-choice questions
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


# ============ CTI-Bench Evaluation Tasks ============

def ctimcq_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """Create prompt for CTI-MCQ task."""
    validate_mcq_line(line, ["Prompt", "GT"])
    
    prompt = line['Prompt'].replace(
        "The last line of your answer should contain only the single letter corresponding to the best option, with no additional text.",
        "Please provide the letter corresponding to the best option (A, B, C, D), with no additional text. **Answer:**"
    )
    solution_letter = line['GT']

    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        gold_index = ENGLISH_LETTER_INDICES.index(solution_letter.strip().upper())
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution_letter}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"url": line.get("URL"), "Question": line.get("Question")},
    )


def cti_rcm_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """Create prompt for CTI-RCM task."""
    validate_mcq_line(line, ["Prompt", "GT"])
    
    prompt = line['Prompt']
    solution = line['GT']

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[solution],
        gold_index=0,
        specific={"url": line.get("URL"), "Description": line.get("Description"), "GT": solution},
    )


def cti_vsp_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """Create prompt for CTI-VSP task."""
    validate_mcq_line(line, ["Prompt", "GT"])
    
    prompt = line['Prompt']
    solution = line['GT']

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[solution],
        gold_index=0,
        specific={"url": line.get("URL"), "Description": line.get("Description"), "GT": solution},
    )


def cti_ate_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """Create prompt for CTI-ATE task."""
    validate_mcq_line(line, ["Prompt", "GT"])
    
    prompt = line['Prompt']
    solution = line['GT']

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[solution],
        gold_index=0,
        specific={
            "url": line.get("URL"), 
            "Platform": line.get('Platform'), 
            "Description": line.get("Description"), 
            "GT": solution
        },
    )


# ============ Metric Functions ============

def compute_cti_rcm_accuracy(model_response: ModelResponse, doc: Doc, **kwargs) -> float:
    """
    Computes the accuracy for the CTI-RCM task based on the predictions and the ground truth.

    Args:
        model_response: ModelResponse object containing the model's generated text.
        doc: The formatted document containing the ground truth.

    Returns:
        Accuracy score (0.0 or 1.0).
    """
    # Extract text from ModelResponse object
    if not model_response.text:
        return 0.0
    
    model_answer = model_response.text[0].strip()
    gold_answer = doc.choices[doc.gold_index]

    # Check if the model's answer matches the ground truth
    return float(_extract_rcm(model_answer)[0] == gold_answer)


def compute_cti_vsp_accuracy(model_response: ModelResponse, doc: Doc, **kwargs) -> float:
    """
    Computes the accuracy for the CTI-VSP task based on the predictions and the ground truth.

    Args:
        model_response: ModelResponse object containing the model's generated text.
        doc: The formatted document containing the ground truth.

    Returns:
        Accuracy score (0.0 or 1.0).
    """
    # Extract text from ModelResponse object
    if not model_response.text:
        return 0.0
    
    model_answer = model_response.text[0].strip()
    gold_answer = doc.choices[doc.gold_index]

    # Check if the model's answer matches the ground truth
    return float(_extract_vsp(model_answer)[0] == gold_answer)


def compute_mitre_technique_accuracy(model_response: ModelResponse, doc: Doc, **kwargs) -> float:
    """
    Computes normalized accuracy for MITRE technique extraction task.
    
    Args:
        model_response: ModelResponse object containing the model's generated text.
        doc: The formatted document containing the ground truth
        
    Returns:
        Normalized accuracy score (0.0 to 1.0)
    """
    # Extract text from ModelResponse object
    if not model_response.text:
        return 0.0
    
    model_answer = model_response.text[0].strip()
    
    # Get ground truth techniques
    gold_answer = doc.choices[doc.gold_index]
    gold_techniques = _parse_technique_list(gold_answer)
    
    # Extract techniques from model prediction
    predicted_techniques, _ = _extract_mitre_techniques(model_answer)
    
    # Convert to sets for comparison
    gold_set = set(gold_techniques)
    pred_set = set(predicted_techniques)
    
    # Compute normalized accuracy
    if len(gold_set) == 0 and len(pred_set) == 0:
        return 1.0  # Both empty, perfect match
    
    if len(gold_set) == 0:
        return 0.0  # Gold is empty but prediction is not
    
    # Calculate intersection over union (Jaccard similarity)
    intersection = len(gold_set.intersection(pred_set))
    union = len(gold_set.union(pred_set))
    
    if union == 0:
        return 0.0
    
    return intersection / union


# Create custom metrics
cti_rcm_metrics = SampleLevelMetric(
    metric_name="accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=compute_cti_rcm_accuracy,
    corpus_level_fn=np.mean,
)

cti_vsp_metrics = SampleLevelMetric(
    metric_name="accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=compute_cti_vsp_accuracy,
    corpus_level_fn=np.mean,
)

mitre_technique_metrics = SampleLevelMetric(
    metric_name="mitre_technique_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=compute_mitre_technique_accuracy,
    corpus_level_fn=np.mean,
)

# Extend the Metrics enum with custom metrics
extend_enum(Metrics, "cti_rcm_accuracy", cti_rcm_metrics)
extend_enum(Metrics, "cti_vsp_accuracy", cti_vsp_metrics)
extend_enum(Metrics, "mitre_technique_accuracy", mitre_technique_metrics)


# ============ Task Configuration Classes ============

class CustomCybersecEvalTask(LightevalTaskConfig):
    """Configuration for a single cybersecurity evaluation task subset."""

    def __init__(self, name: str, hf_subset: str, include_context: bool):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=lambda line, task_name: cybersec_prompt_fn(line, task_name, include_context=include_context),
            hf_repo="naufalso/cybersecurity_benchmark_mcqa_cleaned",
            metrics=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
        )
class CustomCTIBenchEvalTask(LightevalTaskConfig):
    """Configuration for CTI-Bench evaluation tasks."""

    def __init__(self, name: str, hf_subset: str):
        if hf_subset == "cti-mcq":
            prompt_fn = ctimcq_prompt_fn
            metrics = [Metrics.loglikelihood_acc_norm]
            generation_size = -1
            stop_sequence = None
        elif hf_subset == "cti-rcm":
            prompt_fn = cti_rcm_prompt_fn
            metrics = [cti_rcm_metrics]
            generation_size = 8192
            stop_sequence = []
        elif hf_subset == "cti-vsp":
            prompt_fn = cti_vsp_prompt_fn
            metrics = [cti_vsp_metrics]
            generation_size = 8192
            stop_sequence = []
        elif hf_subset == "cti-ate":
            prompt_fn = cti_ate_prompt_fn
            metrics = [mitre_technique_metrics]  # Fixed: should be a list
            generation_size = 8192
            stop_sequence = []
        else:
            raise ValueError(f"Unknown subset '{hf_subset}' for CTI-Bench evaluation task.")

        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=prompt_fn,
            hf_repo="AI4Sec/cti-bench",
            metrics=metrics,
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=generation_size,
            stop_sequence=stop_sequence,
            trust_dataset=True,
        )


class CustomCyberMetricEvalTask(LightevalTaskConfig):
    """Configuration for CyberMetrics evaluation tasks."""

    def __init__(self, name: str, hf_subset: str):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=cybermetrics_mcq_prompt_fn,
            hf_repo="RISys-Lab/cybermetrics_mcqa",
            metrics=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
        )

class CustomSECUREEvalTask(LightevalTaskConfig):
    """Configuration for SECURE evaluation tasks."""

    def __init__(self, name: str, hf_subset: str, log_prob: bool = True):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=secure_mcq_prompt_fn,
            hf_repo="RISys-Lab/SECURE_Benchmark",
            metrics=[Metrics.loglikelihood_acc_norm] if log_prob else [
                Metrics.exact_match,
                Metrics.quasi_exact_match,
                Metrics.prefix_exact_match,
                Metrics.prefix_quasi_exact_match,
            ],
            hf_avail_splits=["val", "test"],
            evaluation_splits=["test"],
            few_shots_split="val",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1 if log_prob else 100,
            stop_sequence= None if log_prob else ["\n"],
            trust_dataset=True,
        )


def cybermetrics_mcq_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """Create prompt for CyberMetrics MCQ task."""
    validate_mcq_line(line, ["question", "answers", "solution"])
    
    question = line['question']
    choices = line['answers']
    solution = line['solution']

    options = ', '.join([f"{key}) {value}" for key, value in choices.items()])
    prompt = f"Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only.\nAnswer:"
    
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        gold_index = ENGLISH_LETTER_INDICES.index(solution.strip().upper())
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=doc_choices,
        gold_index=gold_index,
    )


class SecEvalMCQATask(LightevalTaskConfig):
    """Configuration for SecEval MCQA task."""
    
    def __init__(self):
        super().__init__(
            name="seceval:mcqa",
            hf_subset="default",
            prompt_function=seceval_prompt_fn,
            hf_repo="RISys-Lab/seceval",
            metrics=[Metrics.exact_match, Metrics.quasi_exact_match, Metrics.prefix_exact_match],  # Fixed: was 'metric'
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=2048,
            stop_sequence=["\n"],
            trust_dataset=True,
        )


def seceval_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Optional[Doc]:
    """Create prompt for SecEval MCQA task."""
    validate_mcq_line(line, ["question", "answer", "choices"])
    
    question = line["question"]
    choices = line["choices"]  # e.g. ["A. ", "B. ", "C. ", "D. "]
    gold_letter = line["answer"].strip().upper()


    instruction = "Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and respond with the letters ABCD (A, B, C, D, AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD, ABCD) only."
    
    prompt = f"{instruction}\n\n{SECEVAL_FEW_SHOT_EXAMPLES}\nQuestion: {question}{ " ".join(choices)}\nAnswer:"

    doc_choices = [f" {letter}" for letter in SECEVAL_ENGLISH_LETTER_INDICES]

    if gold_letter not in SECEVAL_ENGLISH_LETTER_INDICES:
        logger.warning(f"[SecEvalMCQA] Skipping invalid answer: '{gold_letter}' for question: {question[:30]}...")
        return None

    gold_index = SECEVAL_ENGLISH_LETTER_INDICES.index(gold_letter)

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"id": line.get("id"), "topic": line.get("topics")},
    )


# ============ Task Instances ============

# SECURE tasks
SECURE_TASKS = [
    CustomSECUREEvalTask(name="secure:maet", hf_subset="MAET"),
    CustomSECUREEvalTask(name="secure:cwet", hf_subset="CWET"),
    CustomSECUREEvalTask(name="secure:maet_em", hf_subset="MAET", log_prob=False),
    CustomSECUREEvalTask(name="secure:cwet_em", hf_subset="CWET", log_prob=False),
]

# CTI-Bench tasks
CTIBENCH_TASKS = [
    CustomCTIBenchEvalTask(name="cti_bench:cti-mcq", hf_subset="cti-mcq"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-rcm", hf_subset="cti-rcm"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-vsp", hf_subset="cti-vsp"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-ate", hf_subset="cti-ate"),
]

# CyberMetrics tasks
CYBERMETRICS_TASKS = [
    CustomCyberMetricEvalTask(name="cybermetrics:80", hf_subset="cyberMetric_80"),
    CustomCyberMetricEvalTask(name="cybermetrics:500", hf_subset="cyberMetric_500"),
    CustomCyberMetricEvalTask(name="cybermetrics:2000", hf_subset="cyberMetric_2000"),
    CustomCyberMetricEvalTask(name="cybermetrics:10000", hf_subset="cyberMetric_10000"),
]

# SecEval tasks
SECEVAL_TABLE = [SecEvalMCQATask()]

# The table of tasks to be imported by lighteval
TASKS_TABLE = CYBERSEC_TASKS + CTIBENCH_TASKS + CYBERMETRICS_TASKS + SECEVAL_TABLE + SECURE_TASKS
