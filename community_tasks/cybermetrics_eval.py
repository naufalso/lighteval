from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Define the English letter indices for choices
ENGLISH_LETTER_INDICES = ["A", "B", "C", "D"]

def cybermetrics_mcq_prompt_fn(line: dict, task_name: str = None) -> Doc:
    question = line['question']
    choices = line['answers']
    solution = line['solution']

    options = ', '.join([f"{key}) {value}" for key, value in choices.items()])

    prompt =  f"Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only.\nAnswer:"
    solution_letter = line['GT']
    
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        # solution_letter is "A", "B", etc. We need its index in ENGLISH_LETTER_INDICES
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
    )

class CustomCyberMetricEvalTask(LightevalTaskConfig):
    """
    Configuration for a single cybersecurity evaluation task subset.
    """

    def __init__(
        self,
        name: str,
        hf_subset: str,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=cybermetrics_mcq_prompt_fn,
            # IMPORTANT: Replace with your actual Hugging Face Hub dataset repository ID
            # For example: "my_organization/my_cybersecurity_dataset"
            hf_repo="RISys-Lab/cybermetrics_mcqa",
            metric=[Metrics.loglikelihood_acc_norm],  # Using standard accuracy for multiple-choice questions
            hf_avail_splits=["train"],  # As per your dataset card
            evaluation_splits=["train"],  # As per your dataset card
            few_shots_split=None,  # No few-shot examples specified
            few_shots_select=None,  # No few-shot selection strategy
            suite=["community"],  # Add this task to the community suite
            generation_size=-1,  # For multiple-choice (loglikelihood) evaluations
            stop_sequence=None,  # Not applicable for non-generative tasks
            trust_dataset=True,  # Default, set to True if you trust the dataset source implicitly
        )

TASKS_TABLE = [
    CustomCyberMetricEvalTask(
        name="cybermetrics:80",
        hf_subset="cyberMetric_80",
    ),
    CustomCyberMetricEvalTask(
        name="cybermetrics:500",
        hf_subset="cyberMetric_500",
    ),
    CustomCyberMetricEvalTask(
        name="cybermetrics:2000",
        hf_subset="cyberMetric_2000",
    ),
    CustomCyberMetricEvalTask(
        name="cybermetrics:10000",
        hf_subset="cyberMetric_10000",
    ),
]