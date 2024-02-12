import uuid
import os

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import DynamicChatPromptBuilder
from haystack import Pipeline
from haystack.dataclasses import ChatMessage

from context_haystack.context import ContextAIAnalytics


model = "gpt-3.5-turbo"
os.environ["GETCONTEXT_TOKEN"] = "GETCONTEXT_TOKEN"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

prompt_builder = DynamicChatPromptBuilder()
llm = OpenAIChatGenerator(model=model)
prompt_analytics = ContextAIAnalytics()
assistant_analytics = ContextAIAnalytics()

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.add_component("prompt_analytics", prompt_analytics)
pipe.add_component("assistant_analytics", assistant_analytics)

pipe.connect("prompt_builder.prompt", "llm.messages")
pipe.connect("prompt_builder.prompt", "prompt_analytics")
pipe.connect("llm.replies", "assistant_analytics")

context_parameters = {"thread_id": uuid.uuid4(), "metadata": {"model": model, "user_id": "1234"}}
location = "Berlin"
messages = [ChatMessage.from_system("Always respond in Englist even if some input data is in other languages."),
            ChatMessage.from_user("Tell me about {{location}}")]

response = pipe.run(
    data={
        "prompt_builder": {"template_variables":{"location": location}, "prompt_source": messages},
        "prompt_analytics": context_parameters,
        "assistant_analytics": context_parameters,
    }
)

print(response)
