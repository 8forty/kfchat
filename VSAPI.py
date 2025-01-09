import logging

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSAPI:
    def __init__(self, api_type: str, parms: dict[str, str]):
        """

        :param api_type: currently: ['azure', 'chroma']
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint...
        """
        if api_type in ['azure', 'chroma']:
            self.api_type = api_type
            self.parms = parms
            self.api_client = None
        else:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type}')

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    def type(self) -> str:
        return self.api_type

    # def client(self) -> openai.OpenAI:
    #     if self.api_client is not None:
    #         return self.api_client
    #
    #     if self.api_type == "azure":
    #         # token_provider = azure.identity.get_bearer_token_provider(
    #         #     azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    #         # )
    #         # client = openai.AzureOpenAI(
    #         #     api_version=self.parms.get("AZURE_OPENAI_API_VERSION"),
    #         #     azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
    #         #     azure_ad_token_provider=token_provider,
    #         # )
    #         log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("AZURE_OPENAI_ENDPOINT")=}, {self.parms.get("AZURE_OPENAI_API_KEY")}, '
    #                  f'{redact(self.parms.get("AZURE_OPENAI_API_VERSION"))}')
    #         self.api_client = openai.AzureOpenAI(azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
    #                                              api_key=self.parms.get("AZURE_OPENAI_API_KEY"),
    #                                              api_version=self.parms.get("AZURE_OPENAI_API_VERSION"))
    #     elif self.api_type == "ollama":
    #         log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("OLLAMA_ENDPOINT")=}')
    #         self.api_client = openai.OpenAI(base_url=self.parms.get("OLLAMA_ENDPOINT"),
    #                                         api_key="nokeyneeded")
    #     elif self.api_type == "openai":
    #         log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("OPENAI_ENDPOINT")}, {redact(self.parms.get("OPENAI_API_KEY"))}')
    #         self.api_client = openai.OpenAI(base_url=self.parms.get("OPENAI_ENDPOINT"),
    #                                         api_key=self.parms.get("OPENAI_API_KEY"))
    #     elif self.api_type == "groq":
    #         log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("GROQ_ENDPOINT")}, {redact(self.parms.get("GROQ_API_KEY"))}')
    #         self.api_client = openai.OpenAI(base_url=self.parms.get("GROQ_OPENAI_ENDPOINT"),
    #                                         api_key=self.parms.get("GROQ_API_KEY"))
    #     elif self.api_type == "github":
    #         base_url = "https://models.inference.ai.azure.com"
    #         log.info(f'building LLM API for [{self.api_type}]: {base_url=}, {redact(self.parms.get("GITHUB_TOKEN"))}')
    #         self.api_client = openai.OpenAI(base_url=base_url,
    #                                         api_key=self.parms.get("GITHUB_TOKEN"))
    #     else:
    #         raise ValueError(f'invalid api_type! {self.api_type}')
    #
    #     return self.api_client

