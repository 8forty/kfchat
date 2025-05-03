import config
from llmconfig.llm_anthropic_config import LLMAnthropicConfig
from llmconfig.llm_ollama_config import LLMOllamaConfig
from llmconfig.llm_openai_config import LLMOpenAIConfig
from llmconfig.llm_openai_ggai_config import LLMOpenAIGGAIConfig
from llmconfig.llmconfig import LLMConfig
from llmconfig.llmsettings import LLMSettings


def create_one(model_spec: config.ModelSpec, settings: dict[str, LLMSettings]) -> LLMConfig:
    """
    creates an LLMConfig of the correct type for the given model_spec
    :rtype: LLMConfig
    """
    # todo: cleanup settings warnings
    if model_spec.api.upper() == 'OPENAI':
        if model_spec.provider.upper() == 'GEMINI':
            return LLMOpenAIGGAIConfig(model_name=model_spec.name, provider=model_spec.provider,
                                       settings=settings[model_spec.api.lower()])
        else:
            return LLMOpenAIConfig(model_name=model_spec.name, provider=model_spec.provider,
                                   settings=settings[model_spec.api.lower()])
    elif model_spec.api.upper() == 'ANTHROPIC':
        return LLMAnthropicConfig(model_name=model_spec.name, provider=model_spec.provider,
                                  settings=settings[model_spec.api.lower()])
    elif model_spec.api.upper() == 'OLLAMA':
        return LLMOllamaConfig(model_name=model_spec.name, provider=model_spec.provider,
                               settings=settings[model_spec.api.lower()])
    else:
        raise ValueError(f'Invalid model spec: {model_spec}')
