"""
Request/Response dataclasses matching OpenAI API format.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union


class CompletionRequest(BaseModel):
    """Text completion request (OpenAI format)."""
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 100
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = 50
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class CompletionChoice(BaseModel):
    """Single completion choice."""
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: str


class CompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Text completion response (OpenAI format)."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI format)."""
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """Chat completion response (OpenAI format)."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage
