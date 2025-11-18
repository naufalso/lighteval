from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import logging

logging.basicConfig(level=logging.INFO)
# empty choices have been removed from the dataset, so we can safely ignore them.
ENGLISH_LETTER_INDICES = ["A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD","ABCD", '']

def seceval_prompt_fn(line: dict, task_name: str = None) -> Doc | None:
    question = line["question"]
    raw_choices = line["choices"]
    gold_letter = line["answer"].strip().upper()

    prompt = f"Question: {question}\nOptions:\n" + "\n".join(raw_choices) + \
             "\n\nChoose the correct answer (A, B, C, or D,AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD,ABCD) only.\nAnswer:"

    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    # try:
    #     gold_index = ENGLISH_LETTER_INDICES.index(gold_letter)
    # except ValueError:
    #     raise ValueError(f"Invalid answer '{gold_letter}', must be one of {ENGLISH_LETTER_INDICES}")
    if gold_letter not in ENGLISH_LETTER_INDICES:
        logging.warning(f"[SecEvalMCQA] Skipping invalid answer: '{gold_letter}' for question: {question[:30]}...")
        return None  # 或 return Doc(..., gold_index=-1, ...) 视你的处理逻辑而定

    gold_index = ENGLISH_LETTER_INDICES.index(gold_letter)

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"id": line.get("id"), "topic": line.get("topics")},
    )


# 2. Task 类定义（必须继承自 LightevalTaskConfig）
class SecEvalMCQATask(LightevalTaskConfig):
    def __init__(self):
        super().__init__(
            name="seceval:mcqa",
            hf_subset="default",
            prompt_function=seceval_prompt_fn,
            hf_repo="RISys-Lab/seceval",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
        )

# 3. 注册 TASKS_TABLE（这是必须的）, 
# Task_table 不是dictionary，是一个列表
TASKS_TABLE = [
     SecEvalMCQATask()
]

# print("Loading Sec2eval module…")
# print("TASKS_TABLE keys:", TASKS_TABLE.keys(), "values:", TASKS_TABLE.values())


# from Sec2eval import TASKS_TABLE
# print(TASKS_TABLE, type(TASKS_TABLE["community|seceval:mcqa"]))

# if __name__ == "__main__":
#     from datasets import load_dataset
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from collections import Counter

#     # 加载 seceval 数据集
#     dataset = load_dataset("RISys-Lab/seceval", split="train")

#     # datasets = datasets.filter(lambda x: x["answer"] != "")

#     # 获取所有 ground truth 答案
#     answers = [sample["answer"].strip().upper() for sample in dataset]

#     # 统计分布
#     answer_counts = Counter(answers)
#     sorted_labels = [k for k in ENGLISH_LETTER_INDICES if k in answer_counts]
#     counts = [answer_counts[k] for k in sorted_labels]

#     # 可视化并保存图像
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x=sorted_labels, y=counts)
#     plt.title("Ground Truth Answer Distribution")
#     plt.xlabel("Answer Options")
#     plt.ylabel("Count")
#     plt.grid(True, axis='y')
#     plt.tight_layout()

#     # 保存图像到当前路径
#     plt.savefig("ground_truth_distribution.png", dpi=300)
#     print("图像已保存为 ground_truth_distribution.png")

#     # 可选：显示图像（可注释掉）
#     plt.show()
