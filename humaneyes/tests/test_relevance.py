import json

import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from humaneyes import eval_model, config


@pytest.fixture
def eval_llm():
    cfg = config.load_config("config.yaml")
    return eval_model.get_deepeval_llm(cfg.ai)


with open("tests/test_data/test_case.json", "r") as file:
    test_cases = json.load(file)

cases = []
for test_case in test_cases:
    cases.append(LLMTestCase(**test_case))

dataset = EvaluationDataset(test_cases=cases)


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    cfg = config.load_config("tests/config.yaml")
    model = eval_model.get_deepeval_llm(cfg.ai)
    hallucination_metric = HallucinationMetric(threshold=0.3, model=model)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=model)
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])
