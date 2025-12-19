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
