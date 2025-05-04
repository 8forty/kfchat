import logging

from ollama import ChatResponse

import logstuff
from llmconfig.llmexchange import LLMExchange, LLMMessagePair
from llmconfig.llmsettings import LLMSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMOllamaExchange(LLMExchange):
    def __init__(self, prompt: str, chat_response: ChatResponse,
                 provider: str,
                 model_name: str,
                 settings: LLMSettings,
                 response_duration_seconds: float):
        stop_problem_reasons = ['max_tokens', 'tool_use']  # todo: no idea if these occur in ollama

        # gemma3:1b prompt:"wake up" sysmsg:""
        # ChatResponse(model='gemma3:1b',
        # created_at='2025-05-04T17:02:29.3114177Z', done=True, done_reason='stop',
        # total_duration=5692294400, load_duration=4930990500,
        # prompt_eval_count=11, prompt_eval_duration=492084400, eval_count=31, eval_duration=262639500,
        # message=Message(role='assistant', content='Good morning! 😊 How are you doing today? \n\nIs there anything you’d like to chat about or anything I can help you with?',
        # images=None, tool_calls=None))

        # ChatResponse(model='gemma3:1b', created_at='2025-05-04T17:42:21.6536352Z', done=True, done_reason='stop',
        # total_duration=7405246600, load_duration=3699688600,
        # prompt_eval_count=31, prompt_eval_duration=266328700, eval_count=399, eval_duration=3407195700,
        # message=Message(role='assistant', content='(Adjusts spectacles, a slow, contemplative gaze sweeping across the digital space)\n\nWell, my dear friend, that’s a question that has haunted astronomers for quite some time, and frankly, it’s one that still holds a certain… awe. We\'re not entirely sure *exactly* how many galaxies there are, and that’s a rather profound and challenging thing to grasp. \n\nThe truth is, we’ve only begun to scratch the surface of this vast cosmic ocean.  The sheer scale of the universe… it’s simply staggering.  \n\nCurrently, scientific estimates place the number of galaxies in the observable universe at… around 2 trillion.  That’s two hundred trillion!  And that number is growing, constantly, as we observe more and more distant galaxies. \n\nHowever, it’s important to understand that “galaxy” isn’t a perfectly defined object. It’s more of a grouping, a collection of stars, gas, dust, and dark matter that orbit a common center of gravity.  \n\nThink of it this way: We’ve cataloged thousands of galaxies – each with its own unique characteristics, its own history, its own swirling arms and collisions. But the *number* of galaxies… that’s a constantly shifting, beautifully complex landscape. \n\nWe’re currently working on refining those estimates, employing incredibly sophisticated methods of observing distant objects and analyzing the light they emit.  It’s a monumental task, but the pursuit of understanding the number of galaxies is a journey driven by our innate curiosity, doesn\'t it?  \n\nIt’s a testament to the incredible, almost unfathomable, scope of the cosmos.  \n\n(Pauses, a quiet smile) \n\nDo you have any further questions, my friend? Perhaps we could consider the concept of "cosmic filaments" and how they connect these galaxies... a rather intriguing thought, wouldn\'t you agree?',
        # images=None, tool_calls=None))

        # todo: add extra response parms as some sort of response metadata?
        super().__init__(prompt=prompt,
                         provider=provider,
                         model_name=model_name,
                         settings=settings,
                         responses=[LLMMessagePair(chat_response.message.role, chat_response.message.content)],
                         input_tokens=chat_response.prompt_eval_count,
                         output_tokens=chat_response.eval_count,
                         response_duration_seconds=response_duration_seconds,
                         problems={-1: chat_response.done_reason} if chat_response.done_reason in stop_problem_reasons else {})
