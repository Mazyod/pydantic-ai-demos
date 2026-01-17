from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_model(
    model_id: str = "qwen/qwen3-4b-thinking-2507",
    base_url: str = "http://localhost:1234/v1",
) -> OpenAIChatModel:
    """Get an OpenAI-compatible model configured for LM Studio."""
    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key="lm-studio",
        ),
    )
