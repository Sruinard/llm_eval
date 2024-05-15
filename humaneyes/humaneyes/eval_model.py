from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from humaneyes import config


class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"


def get_deepeval_llm(cfg: config.AI):
    # Replace these with real values
    custom_model = AzureChatOpenAI(
        openai_api_version=cfg.evaluation.api_version,
        azure_deployment=cfg.evaluation.deployment_name,
        azure_endpoint=cfg.evaluation.endpoint,
        openai_api_key=cfg.evaluation.key,
    )
    return AzureOpenAI(model=custom_model)
