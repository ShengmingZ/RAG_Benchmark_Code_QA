import os
import openai
import backoff

openai.api_key = os.getenv("OPENAI_API_KEY","")


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def chatgpt(prompt, model='gpt-3.5-turbo-1106', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    response = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    outputs = [choice["message"]["content"] for choice in response["choices"]]

    return outputs


if __name__ == "__main__":
    print(chatgpt("hello, world"))