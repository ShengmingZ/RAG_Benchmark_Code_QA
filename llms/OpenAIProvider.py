from typing import Dict, List, Union, Optional
from llms.LLMProvider import LLMProvider
from openai import OpenAI, AsyncOpenAI
import os
import time
import json
import time
import tempfile
import os
from typing import List, Dict, Any, Optional



class OpenAIProvider(LLMProvider):
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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop
        self.is_async = is_async

        # for different providers
        if self.organization == 'openai':
            openai_api_key = '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'
            api_key = openai_api_key
            base_url = "https://api.openai.com/v1"
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
                )
        elif self.organization == 'deepseek':
            # api_key = os.getenv('DEEPSEEK_API_KEY')
            # base_url = "https://api.deepseek.com"
            # use ARK for deepseek
            api_key = os.environ.get("ARK_API_KEY")
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
            if not api_key:
                raise ValueError(
                    "ARK API key not found."
                )
        elif self.organization == 'doubao':
            api_key = os.environ.get("ARK_API_KEY")
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
            if not api_key:
                raise ValueError(
                    "ARK API key not found."
                )
        elif self.organization == 'jieyue':
            api_key = os.environ.get("STEP_API_KEY")
            base_url = "https://api.stepfun.com/v1"
            if not api_key:
                raise ValueError(
                    "STEP_API_KEY not found."
                )
        elif self.organization == 'qwen':
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if not api_key:
                raise ValueError(
                    "DASHSCOPE API key not found."
                )
        elif self.organization == 'zhipu':
            api_key = os.environ.get("Zhipu_API_KEY")
            base_url = "https://open.bigmodel.cn/api/paas/v4/"
            if not api_key:
                raise ValueError(
                    "Zhipu API key not found."
                )
        elif self.organization == 'mistral':
            api_key = os.environ.get('MISTRAL_API_KEY')
            base_url = "https://api.mistral.ai/v1/chat/completions"
            if not api_key:
                raise ValueError(
                    "Mistral API key not found."
                )
        # elif self.organization == 'baichuan':
        #     api_key = os.environ.get()
        #     base_url = "https://api.baichuan-ai.com/v1/chat/completions"
        #     if not api_key:
        #         raise ValueError(
        #             "Baichuan API key not found."
        #         )
        else:
            raise ValueError("Unsupported organization")

        # Create client
        if self.is_async:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )

    def _compose_params(self, prompt):
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.is_async,
        }

        if self.stop is not None:
            params['stop'] = self.stop

        self.prompt_validation(prompt)

        params["messages"] = prompt

        return params

    # async def generate_stream(self, prompt, print_time_to_first_token=True):
    #     params = self._compose_params(prompt)
    #
    #     start_time = time.time() if print_time_to_first_token else None
    #     first_token_received = False
    #
    #     stream = await self.client.chat.completions.create(**params)
    #
    #     async for chunk in stream:
    #         if chunk.choices and hasattr(chunk.choices[0], 'delta'):
    #             content = chunk.choices[0].delta.content
    #             if content is not None:
    #                 # print time to first token, record response latency
    #                 if print_time_to_first_token and not first_token_received:
    #                     print('Time to First Token: ', time.time()-start_time)
    #                     first_token_received = True
    #
    #                 yield content

    def generate(self, prompt, return_type='text', include_logits=True, print_generation_time=True):
        start_time = time.time()

        params = self._compose_params(prompt=prompt)

        if include_logits:
            params['logprobs'] = True

        # Make the API call
        response = self.client.chat.completions.create(**params)

        if print_generation_time: print('Complete response generation time: ', time.time() - start_time)

        if return_type == "raw":
            return response
        elif return_type == "text":
            if include_logits:
                logprobs = [token.logprob for token in response.choices[0].logprobs.content]
                return {
                    "text": response.choices[0].message.content,
                    "logprobs": logprobs
                }
            else:
                return response.choices[0].message.content
        else:  # "full"
            return {
                "text": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason,
                "id": response.id
            }

    def batch_generate(self, prompts: List[List[Dict]], custom_id_prefix="task", return_type='text',
                       include_logits=True, completion_window="24h", poll_interval=60, print_status=True) -> List[Any]:
        """
        Use OpenAI's native Batch API for asynchronous processing of multiple prompts.

        Args:
            prompts: List of prompt strings
            return_type: 'text', 'text_with_logits', 'full', or 'raw'
            include_logits: Whether to include log probabilities
            custom_id_prefix: Prefix for custom IDs in batch requests
            completion_window: Time window for batch completion ("24h" only)
            poll_interval: Seconds between status checks
            print_status: Whether to print progress updates

        Returns:
            List of responses in the same order as input prompts

        Note:
            - 50% cost savings compared to standard API
            - Up to 50,000 requests per batch
            - Max 100MB file size
            - Results within 24 hours
        """
        if not prompts:
            return []

        if len(prompts) > 50000:
            raise ValueError("Batch API supports maximum 50,000 requests per batch")

        # Step 1: Create batch tasks
        tasks = []
        for index, prompt in enumerate(prompts):
            # Compose parameters using existing method
            params = self._compose_params(prompt=prompt)

            # Add logprobs if requested
            if include_logits:
                params['logprobs'] = True

            task = {
                "custom_id": f"{custom_id_prefix}-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": params
            }
            tasks.append(task)

        # Step 2: Create JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
            batch_file_path = f.name

        try:
            if print_status:
                print(f"Created batch file with {len(tasks)} requests")

            # Step 3: Upload file to OpenAI
            with open(batch_file_path, 'rb') as f:
                batch_file = self.client.files.create(file=f, purpose="batch")

            if print_status:
                print(f"Uploaded batch file: {batch_file.id}")

            # Step 4: Create batch job
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window
            )

            if print_status:
                print(f"Created batch job: {batch_job.id}")
                print(f"Status: {batch_job.status}")

            # Step 5: Poll for completion
            while True:
                batch_job = self.client.batches.retrieve(batch_job.id)

                if print_status:
                    print(f"Status: {batch_job.status}")
                    if hasattr(batch_job, 'request_counts'):
                        counts = batch_job.request_counts
                        print(f"Progress: {counts.completed}/{counts.total} completed")

                if batch_job.status == "completed":
                    break
                elif batch_job.status == "failed":
                    error_msg = f"Batch job failed"
                    if hasattr(batch_job, 'errors') and batch_job.errors:
                        error_msg += f": {batch_job.errors}"
                    raise RuntimeError(error_msg)
                elif batch_job.status in ["expired", "cancelled"]:
                    raise RuntimeError(f"Batch job {batch_job.status}")

                time.sleep(poll_interval)

            # Step 6: Download and parse results
            if print_status:
                print("Downloading results...")

            # Get output file content
            output_file_id = batch_job.output_file_id
            output_file = self.client.files.content(output_file_id)

            # Parse results
            results = [None] * len(prompts)  # Pre-allocate to maintain order

            for line in output_file.text.strip().split('\n'):
                if line:
                    result = json.loads(line)

                    # Extract index from custom_id
                    custom_id = result['custom_id']
                    index = int(custom_id.split('-')[-1])

                    # Process response based on return_type
                    if result.get('error'):
                        results[index] = {"error": result['error']}
                    else:
                        response = result['response']['body']
                        processed_result = self._process_batch_response(response, return_type, include_logits)
                        results[index] = processed_result

            if print_status:
                print(f"Batch processing complete: {len(prompts)} requests processed")

            return results

        finally:
            # Clean up temporary file
            try:
                os.unlink(batch_file_path)
            except:
                pass

    def _process_batch_response(self, response, return_type, include_logits):
        """Helper method to process individual batch response based on return_type"""

        if return_type == "raw":
            return response
        elif return_type == "text":
            if include_logits:
                logprobs = [token['logprob'] for token in response['choices'][0]['logprobs']['content']]
                return {
                    "text": response['choices'][0]['message']['content'],
                    "logprobs": logprobs
                }
            else:
                return response['choices'][0]['message']['content']
        else:  # "full"
            logprobs = None
            if include_logits and response['choices'][0].get('logprobs'):
                logprobs = [token['logprob'] for token in response['choices'][0]['logprobs']['content']]
            return {
                "text": response['choices'][0]['message']['content'],
                "model": response['model'],
                "usage": response['usage'],
                "finish_reason": response['choices'][0]['finish_reason'],
                "id": response['id'],
                "logprobs": logprobs
            }
