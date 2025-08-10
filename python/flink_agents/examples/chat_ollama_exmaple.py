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
import os
from typing import Any, Dict, Tuple, Type

from flink_agents.api.agent import Agent
from flink_agents.api.chat_message import ChatMessage, MessageRole
from flink_agents.api.chat_models.chat_model import (
    ChatModelConnection,
    ChatModelSettings,
)
from flink_agents.api.decorators import action, chat_model, chat_model_connection, tool
from flink_agents.api.events.chat_event import ChatRequestEvent, ChatResponseEvent
from flink_agents.api.events.event import (
    InputEvent,
    OutputEvent,
)
from flink_agents.api.execution_environment import AgentsExecutionEnvironment
from flink_agents.api.runner_context import RunnerContext
from flink_agents.integrations.chat_models.ollama_chat_model import (
    OllamaChatModelConnection,
)

model = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:8b")


class MyAgent(Agent):
    """Example agent demonstrating the new ChatModel architecture."""

    # New architecture: separate connection and session
    @chat_model_connection
    @staticmethod
    def ollama_connection() -> Tuple[Type[ChatModelConnection], Dict[str, Any]]:
        """ChatModelConnection responsible for model service connection."""
        return OllamaChatModelConnection, {
            "name": "ollama_connection",
            "model": model,
        }

    @chat_model
    @staticmethod
    def math_chat_model() -> Tuple[Type[ChatModelSettings], Dict[str, Any]]:
        """ChatModelSettings responsible for session configuration, reusing
        connection.
        """
        return ChatModelSettings, {
            "name": "math_chat_model",
            "connection": "ollama_connection",
            "tools": ["add"],
        }

    @tool
    @staticmethod
    def add(a: int, b: int) -> int:
        """Calculate the sum of a and b.

        Parameters
        ----------
        a : int
            The first operand
        b : int
            The second operand

        Returns:
        -------
        int:
            The sum of a and b
        """
        return a + b

    @action(InputEvent)
    @staticmethod
    def process_input(event: InputEvent, ctx: RunnerContext) -> None:
        """User defined action for processing input.

        In this action, we will send ChatRequestEvent to trigger built-in actions.
        """
        input = event.input
        # Use new architecture
        ctx.send_event(
            ChatRequestEvent(
                model="math_chat_model",
                messages=[ChatMessage(role=MessageRole.USER, content=input)],
            )
        )

    @action(ChatResponseEvent)
    @staticmethod
    def process_chat_response(event: ChatResponseEvent, ctx: RunnerContext) -> None:
        """User defined action for processing chat model response."""
        input = event.response
        ctx.send_event(OutputEvent(output=input.content))


# Complete example showing new architecture
class NewArchitectureAgent(Agent):
    """Complete example showing new architecture: one connection, multiple session
    configurations.
    """

    @chat_model_connection
    @staticmethod
    def ollama_connection() -> Tuple[Type[ChatModelConnection], Dict[str, Any]]:
        """Shared model connection."""
        return OllamaChatModelConnection, {
            "name": "ollama_connection",
            "model": model,
        }

    @chat_model
    @staticmethod
    def math_session() -> Tuple[Type[ChatModelSettings], Dict[str, Any]]:
        """Math calculation session."""
        return ChatModelSettings, {
            "name": "math_session",
            "connection": "ollama_connection",
            "tools": ["add"],
        }

    @chat_model
    @staticmethod
    def creative_session() -> Tuple[Type[ChatModelSettings], Dict[str, Any]]:
        """Creative writing session."""
        return ChatModelSettings, {
            "name": "creative_session",
            "connection": "ollama_connection",  # Reuse the same connection
        }

    @tool
    @staticmethod
    def add(a: int, b: int) -> int:
        """Calculate the sum of a and b."""
        return a + b

    @action(InputEvent)
    @staticmethod
    def process_input(event: InputEvent, ctx: RunnerContext) -> None:
        """Choose different sessions based on input content."""
        input_text = event.input.lower()

        if "calculate" in input_text or "sum" in input_text:
            # Use math_session for calculations
            model_name = "math_session"
        else:
            # Use creative_session for other tasks
            model_name = "creative_session"

        ctx.send_event(
            ChatRequestEvent(
                model=model_name,
                messages=[ChatMessage(role=MessageRole.USER, content=event.input)],
            )
        )

    @action(ChatResponseEvent)
    @staticmethod
    def process_chat_response(event: ChatResponseEvent, ctx: RunnerContext) -> None:
        """Process chat response."""
        ctx.send_event(OutputEvent(output=event.response.content))


# Should manually start ollama server before run this example.
if __name__ == "__main__":
    env = AgentsExecutionEnvironment.get_execution_environment()

    input_list = []
    agent = MyAgent()  # Or use NewArchitectureAgent() to test new architecture

    output_list = env.from_list(input_list).apply(agent).to_list()

    input_list.append({"key": "0001", "value": "calculate the sum of 1 and 2."})

    env.execute()

    for key, value in output_list[0].items():
        print(f"{key}: {value}")
