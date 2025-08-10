################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
import uuid
from typing import Any, List, Optional, Sequence, Union

from ollama import ChatResponse, Client, Message
from pydantic import Field

from flink_agents.api.chat_message import ChatMessage, MessageRole
from flink_agents.api.chat_models.chat_model import ChatModelConnection

DEFAULT_CONTEXT_WINDOW = 2048
DEFAULT_REQUEST_TIMEOUT = 30.0


class OllamaChatModelConnection(ChatModelConnection):
    """Ollama ChatModel Connection.

    Visit https://ollama.com/ to download and install Ollama.

    Run `ollama serve` to start a server.

    Run `ollama pull <name>` to download a model to run.
    """

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="Model name to use.")
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
    )
    keep_alive: Optional[Union[float, str]] = Field(
        default="5m",
        description="controls how long the model will stay loaded into memory following the request(default: 5m)",
    )

    __client: Client = None

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
        keep_alive: Optional[Union[float, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Init method."""
        super().__init__(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout,
            keep_alive=keep_alive,
            **kwargs,
        )

    @property
    def client(self) -> Client:
        """Return ollama client."""
        if self.__client is None:
            self.__client = Client(host=self.base_url, timeout=self.request_timeout)
        return self.__client

    def chat(self, messages: Sequence[ChatMessage], tools: Optional[List] = None, **kwargs: Any) -> ChatMessage:
        """Direct communication with Ollama service."""
        # Convert message format
        ollama_messages = self.__convert_to_ollama_messages(messages)

        # Convert tool format
        ollama_tools = None
        if tools is not None:
            ollama_tools = [tool.metadata.to_openai_tool() for tool in tools]

        # Call Ollama API
        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=False,
            tools=ollama_tools,
            options=kwargs,
            keep_alive=self.keep_alive,
        )

        # Convert response format
        return self.__convert_from_ollama_response(response)

    @staticmethod
    def __convert_to_ollama_messages(messages: Sequence[ChatMessage]) -> List[Message]:
        """Convert ChatMessage to Ollama Message format."""
        ollama_messages = [
            Message(
                role=msg.role.value,
                content=msg.content,
            )
            for msg in messages]
        return ollama_messages

    @staticmethod
    def __convert_from_ollama_response(response: ChatResponse) -> ChatMessage:
        """Convert Ollama response to ChatMessage format."""
        ollama_tool_calls = response.message.tool_calls
        if ollama_tool_calls is None:
            ollama_tool_calls = []
        tool_calls = []
        for ollama_tool_call in ollama_tool_calls:
            tool_call = {
                "id": uuid.uuid4(),
                "type": "function",
                "function": {
                    "name": ollama_tool_call.function.name,
                    "arguments": ollama_tool_call.function.arguments,
                },
            }
            tool_calls.append(tool_call)
        return ChatMessage(
            role=MessageRole(response.message.role),
            content=response.message.content,
            tool_calls=tool_calls,
        )
