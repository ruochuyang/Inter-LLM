import time
from collections import Counter, defaultdict
from functools import lru_cache
from pprint import pformat, pprint
from typing import List, Tuple, Union
import numpy as np
from openai import OpenAI, OpenAIError
from pygments import highlight
from pygments.formatters import Terminal256Formatter, TerminalFormatter
from pygments.lexers import PythonLexer

import utils


### Initialize the OpenAI Key ###
config = utils.parse_config("config.yaml")
client = OpenAI(api_key=config["OPENAI_API_KEY"])


def pprint_color(obj, style="staroffice", width=200):
    txt = highlight(
        pformat(obj, width=width), PythonLexer(), Terminal256Formatter(style=style)
    )
    print(txt, end="")
    return txt


class Conversation:
    def __init__(
        self, messages: List[dict], include_env_messages: bool = False
    ) -> None:
        self._messages = messages
        self._include_env_messages = include_env_messages

    def add_message(self, message: dict):
        self._messages.append(message)

    @property
    def messages(self):
        if self._include_env_messages:
            return self._messages
        else:
            return [
                m
                for m in self._messages
                if m["role"].lower() not in ["env", "environment"]
            ]

    @property
    def messages_including_env(self):
        return self._messages


@lru_cache(maxsize=None)
def _send_query_cached(messages: list, model: str, temperature: float):

    assert (
        temperature == 0.0
    ), "Caching only works for temperature=0.0, as otherwise we want to get different responses back"

    messages = [dict(m) for m in messages]

    # print('send_query_cached,', messages)
    # input('wait')

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def _send_query(messages: list, model: str, temperature: float):

    if temperature == 0.0:

        hashable_messages = tuple(tuple(m.items()) for m in messages)

        return _send_query_cached(
            messages=hashable_messages, model=model, temperature=temperature
        )

    else:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )


def _send_query_structured_outputs(messages, model, temperature, response_format):
    # Ensure that LLM outputs adhere to a JSON schema

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
    )

    return completion


class LLM:
    def __init__(self, model: str, temperature: float, debug=False) -> None:

        available_models = [m.id for m in client.models.list().data]
        assert model in available_models, available_models
        self.model = model

        self.temperature = temperature
        self.debug = debug

    def send_query(self, conversation: Conversation, response_format):
        """Example input:
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Knock knock."},
            {"role": "assistant", "content": "Who's there?"},
            {"role": "user", "content": "Orange."},
        ],
        """
        num_attempts = 0

        while True:
            try:
                if response_format is None:
                    # Normal chat completion without structured outputs
                    completion = _send_query(
                        model=self.model,
                        temperature=self.temperature,
                        messages=conversation.messages,
                    )

                    role = completion.choices[0].message.role
                    response = completion.choices[0].message.content

                else:
                    # Structured output format, e.g., JSON format
                    completion = _send_query_structured_outputs(
                        model=self.model,
                        temperature=self.temperature,
                        messages=conversation.messages,
                        response_format=response_format,
                    )

                    role = completion.choices[0].message.role
                    response = completion.choices[0].message.parsed

                break

            except OpenAIError as e:
                print(f"Attempting again after {e}")
                num_attempts += 1
                time.sleep(5)

            assert num_attempts < 10, "Too many OpenAI errors"

        if self.debug:
            pprint_color("#################################\n", width=200)
            pprint_color(
                "\n+++++++++++++++++++++++++++++++++\n".join(
                    [f"{m['role']}: {m['content']}" for m in conversation.messages]
                ),
                width=200,
            )
            pprint_color(
                f"+++++++++++++++++++++++++++++++++\n {role}: {response}",
                width=200,
                style="rrt",
            )

        conversation.add_message({"role": role, "content": response})

        return response
