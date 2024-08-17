import json
import time
import traceback
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

GPT_MODEL_TYPE_DICT = {
    "gpt-3.5-turbo": "chat",
    "gpt-4": "chat",
}

GPT_MODEL_MAX_TOKEN_DICT = {
    "gpt-3.5-turbo": 16000,
    "gpt-4": 8192,
}

# per thousand token cost. 04-11
GPT_MODEL_MAX_COST_DICT = {
    "gpt-3.5-turbo": {
        'input': 0.0005,
        'output': 0.0015,
    },
    "gpt-4": {
        'input': 0.03,
        'output': 0.06
    },
}


def estimate_cost(usage, engine):
    """estimate the cost for input prompt based on the num of token."""
    return (usage.prompt_tokens / 1000.0 * GPT_MODEL_MAX_COST_DICT[engine]['input'] +
            usage.completion_tokens / 1000.0 * GPT_MODEL_MAX_COST_DICT[engine]['output'])


@retry(wait=wait_random_exponential(multiplier=1, max=60))
def call_openai_engine(engine='gpt-3.5-turbo',
                       prompt='',
                       temperature=0.0,
                       api_key='',
                       max_tokens=3000,
                       stop_tokens=["<|endoftext|>"], ):
    try:
        if api_key == '':
            raise ValueError("OpenAI api_key should not be empty!")
        client = openai.OpenAI(api_key=api_key)
        if GPT_MODEL_TYPE_DICT[engine] == "text":
            sample = client.completions.create.create(
                engine=engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120, 
                # stop=stop_tokens,
            )
        elif GPT_MODEL_TYPE_DICT[engine] == "chat":
            sample = client.chat.completions.create(
                model=engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
                # stop=stop_tokens,
            )
        else:
            raise ValueError("Unknown GPT model type")

    except:
        traceback.print_exc()
        print(f'Retrying querying OpenAI {engine}...')

        time.sleep(0.1)

    return sample


def get_openai_generation(engine, openai_api_response):
    """
    Extraction GPT output from OpenAI API response
    """
    if GPT_MODEL_TYPE_DICT[engine] == 'text':
        """An example completion API response looks as follows:
        {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "text_completion",
            "created": 1589478378,
            "model": "text-davinci-003",
            "choices": [
                {
                    "text": "\n\nThis is indeed a test",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12
            }
        }
        """
        generated_content = openai_api_response.choices[0].text
        usage = openai_api_response.usage
    elif GPT_MODEL_TYPE_DICT[engine] == 'chat':
        """An example chat API response looks as follows:
            {
                'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
                'object': 'chat.completion',
                'created': 1677649420,
                'model': 'gpt-3.5-turbo',
                'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'The 2020 World Series was played in Arlington, Texas at the Globe Life Field, which was the new home stadium for the Texas Rangers.'},
                        'finish_reason': 'stop',
                        'index': 0
                    }
                ]
            }
        """
        generated_content = openai_api_response.choices[0].message.content
        usage = openai_api_response.usage
    else:
        raise ValueError('GPT_MODEL_TYPE_DICT[engine] should be text or chat')
    return generated_content, usage
