import dataclasses
import yaml


@dataclasses.dataclass
class AzureAIConfig:
    api_version: str
    endpoint: str
    key: str
    deployment_name: str


@dataclasses.dataclass
class AI:
    evaluation: AzureAIConfig


@dataclasses.dataclass
class Config:
    version: str
    ai: AI


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    return Config(
        version=cfg["version"],
        ai=AI(evaluation=AzureAIConfig(**cfg["ai"]["evaluation"])),
    )
