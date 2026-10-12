"""Data models for function calling input and output using Pydantic."""
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class FunctionParameter(BaseModel):
    """Schema for a single function parameter."""
    type: str


class FunctionDefinition(BaseModel):
    """Schema for a function definition provided to the LLM."""
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: Dict[str, str]


class PromptInput(BaseModel):
    """Schema for the input prompt."""
    prompt: str


class FunctionCallOutput(BaseModel):
    """Schema for the final generated output."""
    prompt: str
    name: str
    parameters: Dict[str, Any]