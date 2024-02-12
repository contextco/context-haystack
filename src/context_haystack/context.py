from typing import List
from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from getcontext.token import Credential
import getcontext.generated as getcontext
from getcontext.generated.models import Message, Thread


@component
class ContextAIAnalytics:
    """
    A component for logging a conversation to the Context API.

    This component logs a conversation to Context.ai, which can be used to analyze and understand the conversation.
    """
    def __init__(
        self,
        auth_token: str = Secret.from_env_var("GETCONTEXT_TOKEN")
    ):
        """
        Create a new ContextAI component.

        :param auth_token: The token to authenticate with the Context API. If not provided, it will be read from the GETCONTEXT_TOKEN environment variable. You can generate a new token at https://with.context.ai/settings/tokens
        """
        self.context_api = getcontext.ContextAPI(credential=Credential(auth_token.resolve_value()))

    @component.output_types(messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage], thread_id: str, metadata: dict = None):
        metadata = {} if metadata is None else metadata
        context_messages = [Message(message=t.content, role=t.role.value) for t in messages]

        thread = Thread(id=thread_id, messages=context_messages, metadata=metadata)
        self.context_api.log.conversation_thread(body={"conversation": thread})

        return {'messages': messages}
