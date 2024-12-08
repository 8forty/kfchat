import openai


class ModelAPI:
    def __init__(self, api_type: str, parms: dict[str, str]):
        if api_type in ['azure', 'ollama', 'openai']:
            self.api_type = api_type
            self.parms = parms
            self.api_client = None
        else:
            raise ValueError(f'invalid api_type! {api_type}')

    def type(self) -> str:
        return self.api_type

    def client(self) -> openai.OpenAI:
        if self.api_client is not None:
            return self.api_client

        if self.api_type == "azure":
            # token_provider = azure.identity.get_bearer_token_provider(
            #     azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            # )
            # client = openai.AzureOpenAI(
            #     api_version=self.parms.get("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
            #     azure_ad_token_provider=token_provider,
            # )
            self.api_client = openai.AzureOpenAI(azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
                                                 api_key=self.parms.get("AZURE_OPENAI_API_KEY"),
                                                 api_version=self.parms.get("AZURE_OPENAI_API_VERSION"))
        elif self.api_type == "ollama":
            self.api_client = openai.OpenAI(
                base_url=self.parms.get("OLLAMA_ENDPOINT"),
                api_key="nokeyneeded",
            )
        elif self.api_type == "openai":
            self.api_client = openai.OpenAI(api_key=self.parms.get("OPENAI_API_KEY"))
        elif self.api_type == "github":
            self.api_client = openai.OpenAI(base_url="https://models.inference.ai.azure.com", api_key=self.parms.get("GITHUB_TOKEN"))
        else:
            raise ValueError(f'invalid api_type! {self.api_type}')

        return self.api_client
