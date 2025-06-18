"""
Recommend config for LLMProviders
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
from pathlib import Path

@dataclass
class LLMConfig:
    organization: str
    model: str
    temperature: float
    max_tokens: int = 1000
    is_async: bool = True               # async or not, default use async
    stop: Optional[List[str]] = None    # add stop sequence, default is no stop
    top_p: float = None                 # ONLY FOR Ollama
    n_gpu_layers: int = None            # ONLY FOR Ollama
    mmap: bool = None                   # ONLY FOR Ollama

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class LLMSettings:
    class OpenAIConfigs:
        openai_new = LLMConfig(organization="openai",
                               model="gpt-4o-mini",
                               temperature=0,
                               max_tokens=500,
                               is_async=False)

        openai_old = LLMConfig(organization='openai',
                               model='gpt-3.5-turbo-0125',
                               temperature=0,
                               max_tokens=500,
                               is_async=False)

    class LLAMAConfigs:
        llama_old_code = LLMConfig(organization='llama',
                                   model='codellama/CodeLlama-13b-Instruct-hf',
                                   temperature=0,
                                   max_tokens=500,
                                   is_async=False)

        llama_old_qa = LLMConfig(organization='llama',
                                 model='meta-llama/Llama-2-13b-chat-hf',
                                 temperature=0,
                                 max_tokens=200,
                                 is_async=False)

        llama_new = LLMConfig(organization='llama',
                              model='meta-llama/Llama-3.1-8B-Instruct',
                              temperature=0,
                              max_tokens=500,
                              is_async=False)

    # class AnthropicConfigs:
    #     anthropic_full = LLMConfig(organization="anthropic",
    #                                model="claude-3-7-sonnet-latest",
    #                                temperature=0.6)
    #
    #     anthropic_light = LLMConfig(organization="anthropic",
    #                                 model="claude-3-5-haiku-latest",
    #                                 temperature=0.6)
    #
    # class GoogleConfigs:
    #     google_full = LLMConfig(organization="google",
    #                             model="gemini-2.5-pro",
    #                             temperature=0.6)
    #
    #     google_light = LLMConfig(organization="google",
    #                              model="gemini-2.0-flash",
    #                              temperature=0.6)
    #
    # class MistralConfigs:
    #     mistral_full = LLMConfig(organization="mistral",
    #                              model="mistral-large-latest",
    #                              temperature=0.6)
    #
    #     mistral_light = LLMConfig(organization="mistral",
    #                               model="ministral-8b-latest",
    #                               temperature=0.6)
    #
    # class QWENConfigs:
    #     local_qwen_full = LLMConfig(organization="ollama",
    #                                 model="qwen2.5:72b",
    #                                 temperature=0.6)
    #
    #     local_qwen_light = LLMConfig(organization="ollama",
    #                                  model="qwen2.5:7b",
    #                                  temperature=0.6)
    #
    #     qwen_full = LLMConfig(organization="qwen",
    #                           model="qwen-max-latest",
    #                           temperature=0.6)
    #
    # class DeepSeekConfigs:
    #     local_deepseek_full = LLMConfig(organization="ollama",
    #                                     model="deepseek-r1:70b",
    #                                     temperature=0.6)
    #
    #     local_deepseek_light = LLMConfig(organization="ollama",
    #                                      model="deepseek-r1:7b",
    #                                      temperature=0.6)
    #
    #     deepseek_full = LLMConfig(organization="deepseek",
    #                               model="deepseek-r1-250120",
    #                               temperature=0.6)
    #
    #     deepseek_light = LLMConfig(organization="deepseek",
    #                                model="deepseek-v3-241226",
    #                                temperature=0.6)
    #
    # class EmoLLMConfigs:
    #     local_EmoLLM_internlm = LLMConfig(organization="ollama",
    #                                       model="emollm-internlm25-7b",
    #                                       temperature=0.6,
    #                                       top_p=0.9,
    #                                       n_gpu_layers=35,
    #                                       mmap=True)
    #
    # class DoubaoConfigs:
    #     doubao_full = LLMConfig(organization="doubao",
    #                        model="doubao-1-5-pro-256k-250115",
    #                        temperature=0.6)
    #     doubao_light = LLMConfig(organization="doubao",
    #                             model="doubao-1-5-lite-32k-250115",
    #                             temperature=0.6)
    #
    # class ZhipuConfigs:
    #     zhipu_full = LLMConfig(organization='zhipu',
    #                            model='glm-4-plus',
    #                            temperature=0.6)
    #
    # class JieyueConfigs:
    #     jieyue_full = LLMConfig(organization='jieyue',
    #                             model='step-1-32k',
    #                             temperature=0.6)
    #
    # class BaichuanConfigs:
    #     baichuan_full = LLMConfig(organization="baichuan",
    #                               model="baichuan2-turbo",
    #                               temperature=0.6)

