import os
import time
import torch
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from llms.LLMProvider import LLMProvider


class LlamaProvider(LLMProvider):
    def __init__(
            self,
            organization,
            model,
            temperature,
            max_tokens,
            is_async,
            stop: List[str] = None,
    ):
        self.organization = organization
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.is_async = is_async

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")

        # Load model and tokenizer
        print(f"📥 Loading model: {self.model_name}")
        access_token = "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"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=torch.float16, token=access_token)

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _compose_params(self, prompt):
        """Convert prompt to text format"""
        if isinstance(prompt, list):
            # Convert messages to chat format
            text = ""
            for msg in prompt:
                if msg['role'] == 'system':
                    text += f"<|system|>\n{msg['content']}\n"
                elif msg['role'] == 'user':
                    text += f"<|user|>\n{msg['content']}\n"
                elif msg['role'] == 'assistant':
                    text += f"<|assistant|>\n{msg['content']}\n"
            text += "<|assistant|>\n"
        else:
            text = prompt

        return text

    def generate_batch(self, prompts, return_type='text', include_logits=True, print_generation_time=True):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        print(f"✅ Model loaded successfully")

        results = list()
        for prompt in prompts:
            result = self.generate(prompt=prompt, return_type=return_type, include_logits=include_logits, print_generation_time=print_generation_time)
            print(result['text'])
            results.append(result)
        return results

    def generate(self, prompt, return_type='text', include_logits=True, print_generation_time=True):
        start_time = time.time()

        # Prepare input
        input_text = self._compose_params(prompt)

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": True if self.temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": include_logits,
        }

        # Add stop tokens if provided
        if self.stop:
            # stop_token_ids = []
            # for stop_word in self.stop:
            #     stop_ids = self.tokenizer.encode(stop_word, add_special_tokens=False)
            #     stop_token_ids.extend(stop_ids)
            # if stop_token_ids:
            #     generation_kwargs["eos_token_id"] = stop_token_ids
            generation_kwargs["stop_strings"] = self.stop

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                tokenizer=self.tokenizer,
                **generation_kwargs
            )

        # Extract generated tokens (remove input tokens)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[0][input_length:]

        # Decode response
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Process logprobs if requested
        logprobs = []
        if include_logits and outputs.scores:
            for score in outputs.scores:
                # Get probabilities and convert to logprobs
                probs = torch.softmax(score[0], dim=-1)
                # Get logprob of the selected token
                selected_token_id = generated_tokens[len(logprobs)] if len(logprobs) < len(generated_tokens) else \
                generated_tokens[-1]
                logprob = torch.log(probs[selected_token_id]).item()
                logprobs.append(logprob)

        if print_generation_time:
            print('Complete response generation time: ', time.time() - start_time)

        # Return based on return_type
        if return_type == "raw":
            return {
                "sequences": outputs.sequences,
                "scores": outputs.scores if include_logits else None,
                "input_length": input_length,
                "response_text": response_text
            }
        elif return_type == "text":
            if include_logits:
                return {
                    "text": response_text,
                    "logprobs": logprobs
                }
            else:
                return response_text
        else:  # "full"
            return {
                "text": response_text,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": input_length + len(generated_tokens)
                },
                "finish_reason": "stop",
                "id": f"llama-{int(time.time())}",
                "logprobs": logprobs if include_logits else []
            }