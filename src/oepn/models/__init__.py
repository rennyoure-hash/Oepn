from oepn.core.model import BaseModel, ModelResponse


def infer_model(model_str: str) -> BaseModel:
    from oepn.models.openai import OpenAIModel
    from oepn.models.anthropic import AnthropicModel
    from oepn.models.gemini import GeminiModel
    from oepn.models.litellm import LiteLLMModel

    if ":" in model_str:
        provider, model_name = model_str.split(":", 1)
    else:
        provider = ""
        model_name = model_str

    provider_map = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "gemini": GeminiModel,
        "google": GeminiModel,
    }

    cls = provider_map.get(provider)
    if cls is not None:
        return cls(model_id=model_name)

    return LiteLLMModel(model_id=model_str)
