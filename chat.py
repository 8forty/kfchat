import logging
import sys
from dataclasses import dataclass

import dotenv
import openai
from openai.types.chat import ChatCompletion

import config
import logstuff
from modelapi import ModelAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class ChatExchange:
    prompt: str
    response: str


class Chat:

    def __init__(self, api_type: str):
        self.env: dict[str, str] = dotenv.dotenv_values()
        self.model_api: ModelAPI = ModelAPI(api_type, parms=self.env)
        self.client: openai.OpenAI = self.model_api.client()

    def model_api_type(self):
        return self.model_api.type()

    @staticmethod
    def check_for_stop_problems(response: ChatCompletion) -> dict[int, str]:
        stop_problems: dict[int, str] = {}
        for i in range(0, len(response.choices)):
            choice = response.choices[i]
            stop_problem = ''
            match choice.finish_reason:
                case 'length':
                    stop_problem = 'too many tokens in request'
                case 'content_filter':
                    stop_problem = 'flagged by content filter'
                case 'tool_calls':
                    stop_problem = 'called a tool'
                case 'function_call':
                    stop_problem = 'called a function'

            if len(stop_problem) > 0:
                stop_problems[i] = stop_problem  # add problem to the dict

        return stop_problems

    def chat(self, model_name: str, temp: float, max_tokens: int, n: int,
             sysmsg: str, prompt: str, convo: list[ChatExchange]):
        # todo: detect stop problems, e.g. not enough tokens
        messages = [{'role': 'system', 'content': sysmsg}]
        for exchange in convo:
            messages.append({'role': 'user', 'content': exchange.prompt})
            messages.append({'role': 'assistant', 'content': exchange.response})
        messages.append({'role': 'user', 'content': prompt})

        response: ChatCompletion = self.client.chat.completions.create(
            model=model_name,
            temperature=temp,  # default 1.0, 0.0->2.0
            messages=messages,
            max_tokens=max_tokens,  # default 16?
            n=n,

            stream=False,

            # seed=27,
            # top_p=1,  # default 1, ~0.01->1.0
            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],
        )

        # ChatCompletion (full openai version):
        #     id: str  A unique identifier for the chat completion
        #     choices: List[Choice]  A list of chat completion choices. Can be more than one if `n` is greater than 1.
        #     created: int  The Unix timestamp (in seconds) of when the chat completion was created.
        #     model: str  The model used for the chat completion.
        #     object: Literal["chat.completion"]  The object type, which is always `chat.completion`
        #     service_tier: Optional[Literal["scale", "default"]] = None  The service tier used, included iff the `service_tier` parameter is specified in the request.
        #     system_fingerprint: Optional[str] = None  This fingerprint represents the backend configuration that the model runs with, in conjunction with the `seed` to detect backend changes
        #     usage: Optional[CompletionUsage] = None  Usage statistics for the completion request.

        for choice_idx, stop_problem in self.check_for_stop_problems(response).items():
            log.warning(f'** stop problem choice[{choice_idx}]! {stop_problem}**\n')

        return response
