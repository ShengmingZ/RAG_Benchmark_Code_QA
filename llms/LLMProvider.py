from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional


class LLMProvider(ABC):
    """
    LLM provider for different API.
    organization:
    model: claude model name
    temperature:
    max_tokens:
    is_async: whether to use async streaming or not
    stop: a list of stop tokens
    """
    @abstractmethod
    def __init__(
            self,
            organization: str,
            model: str,
            temperature: float,
            max_tokens: int,
            is_async: bool,
            stop: List[str],
            **kwargs
    ):
        pass

    @staticmethod
    def prompt_validation(prompt):
        # validate the format of prompt
        if isinstance(prompt, str):
            ...
        else:
            assert prompt[0]['role'] in ['system', 'user'], print('first message must be system or user, not {}'.format(prompt))
            assert prompt[-1]['role'] == 'user', print('last message must be user')
            for i in range(1, len(prompt)):
                if prompt[i]['role'] not in ['assistant', 'user']:
                    raise Exception(f'invalid role in prompt: {prompt[i]["role"]}')
                if prompt[i]['role'] == 'assistant' and prompt[i - 1]['role'] != 'user':
                    raise Exception('invalid prompt format: assistant message should follow user message')
        return True

    # @abstractmethod
    # async def generate_stream(self, prompt, **kwargs):
    #     """
    #     async generate
    #     prompt: str ot dict, str indicates single user prompt, dict should be in the format of {"role": , "content": },
    #             role in ('system', 'user', 'assistant'), and the sequence must be 'system', 'user', 'assistant', 'user', ...
    #     """
    #     pass
    #
    # @abstractmethod
    # def generate(self, prompt, **kwargs):
    #     pass




