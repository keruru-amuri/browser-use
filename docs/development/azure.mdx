# Adapting Microsoft Foundry SDK (Azure AI Inference) to Browser Use

This guide explains how to adapt the `browser-use` codebase to use the Microsoft Foundry SDK (specifically `azure-ai-inference`) for consuming models from the Azure AI Model Catalog (e.g., Llama 3, Phi-3, Mistral).

## Prerequisites

1.  **Install the SDK**:
    You need to install the `azure-ai-inference` package.
    ```bash
    pip install azure-ai-inference
    # or with uv
    uv add azure-ai-inference
    ```

2.  **Get Credentials**:
    Obtain your endpoint URL and API key from the Azure AI Foundry portal (or GitHub Models).

## Implementation Steps

### 1. Create the Integration Module

Create a new directory `browser_use/llm/azure_foundry` and add a `chat.py` file.

**File:** `browser_use/llm/azure_foundry/chat.py`

```python
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

@dataclass
class ChatAzureFoundry(BaseChatModel):
    """
    Integration for Microsoft Foundry SDK (Azure AI Inference).
    """
    model: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1000
    
    _client: Optional[ChatCompletionsClient] = field(default=None, init=False)

    def __post_init__(self):
        self.endpoint = self.endpoint or os.getenv("AZURE_INFERENCE_ENDPOINT")
        self.api_key = self.api_key or os.getenv("AZURE_INFERENCE_CREDENTIAL")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure Inference Endpoint and API Key must be provided.")

    @property
    def provider(self) -> str:
        return "azure_foundry"

    def get_client(self) -> ChatCompletionsClient:
        if not self._client:
            self._client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._client

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        output_format: Any = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        client = self.get_client()
        
        # Convert messages to Azure AI Inference format
        formatted_messages = []
        for msg in messages:
            if msg.role == "system":
                formatted_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                formatted_messages.append(UserMessage(content=msg.content))
            elif msg.role == "assistant":
                formatted_messages.append(AssistantMessage(content=msg.content))

        # Call the API
        response = await client.complete(
            messages=formatted_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )

        # Extract content and usage
        content = response.choices[0].message.content
        usage = ChatInvokeUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )

        return ChatInvokeCompletion(
            completion=content,
            usage=usage,
        )
```

### 2. Register the New Model

Update `browser_use/llm/__init__.py` to export the new class.

**File:** `browser_use/llm/__init__.py`

```python
# ... existing imports ...
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ... existing type hints ...
    from browser_use.llm.azure_foundry.chat import ChatAzureFoundry

# ... existing lazy imports ...
_LAZY_IMPORTS = {
    # ... existing mappings ...
    'ChatAzureFoundry': ('browser_use.llm.azure_foundry.chat', 'ChatAzureFoundry'),
}

__all__ = [
    # ... existing exports ...
    'ChatAzureFoundry',
]
```

### 3. Usage Example

```python
import asyncio
from browser_use import Agent, Browser
from browser_use.llm.azure_foundry.chat import ChatAzureFoundry

async def main():
    llm = ChatAzureFoundry(
        model="Phi-3-medium-4k-instruct",
        endpoint="https://<your-endpoint>.services.ai.azure.com/models",
        api_key="<your-api-key>"
    )
    
    agent = Agent(
        task="Go to google.com and search for 'Microsoft Foundry SDK'",
        llm=llm,
        browser=Browser()
    )
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. (Optional) Add Convenience Accessors

To use models like `llm.azure_foundry_phi_3` directly, update `browser_use/llm/models.py`.

**File:** `browser_use/llm/models.py`

1.  Import the new class (inside `TYPE_CHECKING` block and `__getattr__`).
2.  Update `get_llm_by_name` to handle `azure_foundry` provider.

```python
# In get_llm_by_name function:
    # ... existing providers ...
    elif provider == 'azure_foundry':
        # Example: azure_foundry_phi_3_mini
        # You might need to map the model name to the actual deployment name or model ID
        return ChatAzureFoundry(model=model_part.replace('_', '-'))
```

## Notes

*   **Structured Output**: The `azure-ai-inference` SDK supports structured output (JSON mode) for some models. You may need to adapt the `ainvoke` method to handle `output_format` by passing `response_format={"type": "json_object"}` if supported by the model.
*   **Image Support**: If using multimodal models (like GPT-4o or Llama 3.2 Vision), you will need to handle image content parts in the message conversion logic.
