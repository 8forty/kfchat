import config
from llmconfig.llm_anthropic_config import LLMAnthropicConfig
from llmconfig.llm_openai_config import LLMOpenAIConfig
from llmconfig.llm_openai_ggai_config import LLMOpenAIGGAIConfig
from llmconfig.llmconfig import LLMConfig
from llmconfig.llmsettings import LLMSettings


def create_one(model_spec: config.ModelSpec, settings: dict[str, LLMSettings]) -> LLMConfig:
    # todo: cleanup settings warnings
    if model_spec.api.lower() == 'openai':
        if model_spec.provider.upper() == 'GEMINI':
            return LLMOpenAIGGAIConfig(model_name=model_spec.name, provider=model_spec.provider, settings=settings[model_spec.api.lower()])
        else:
            return LLMOpenAIConfig(model_name=model_spec.name, provider=model_spec.provider, settings=settings[model_spec.api.lower()])
    elif model_spec.api.lower() == 'anthropic':
        return LLMAnthropicConfig(model_name=model_spec.name, provider=model_spec.provider, settings=settings[model_spec.api.lower()])
    else:
        raise ValueError(f'Invalid model spec: {model_spec}')
