## Evaluate LLM application

1. install poetry
2. create virtual env
3. from [humaneyes](./humaneyes) run: `poetry install`
4. fill out [config.yaml.example](config.yaml.example) and copy it to `./tests/config.yaml`
5. run: `pytest tests`


for some nicer results:
1. from [humaneyes](./humaneyes) folder run: `deepeval test run tests/test_relevance.py`
