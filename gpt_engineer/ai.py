from __future__ import annotations

import logging

import openai
from langchain import PromptTemplate, LLMChain HuggingFacePipeline

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])


question = "What is electroencephalography?"


logger = logging.getLogger(__name__)


class AI:
    def __init__(self, model="gpt-4", temperature=0.1):
        self.llm = HuggingFacePipeline.from_model_id(
    model_id="mosaicml/mpt-7b-instruct",
    task="text-generation",
    model_kwargs={"temperature": temperature, "max_length": 2048},
)
        # self.temperature = temperature
        
        # self.model = model

    def start(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return self.next(messages)

    def fsystem(self, msg):
        return {"role": "system", "content": msg}

    def fuser(self, msg):
        return {"role": "user", "content": msg}

    def fassistant(self, msg):
        return {"role": "assistant", "content": msg}

    def next(self, messages: list[dict[str, str]], prompt=None):
        if prompt:
            messages += [{"role": "user", "content": prompt}]
        print(f"Creating a new chat completion: {messages}")
        
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run(messages)
        print("responsed from llm", response)

        logger.debug(f"Creating a new chat completion: {messages}")
        # response = openai.ChatCompletion.create(
        #     messages=messages,
        #     stream=True,
        #     model=self.model,
        #     temperature=self.temperature,
        # )

        chat = []
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            msg = delta.get("content", "")
            print(msg, end="")
            chat.append(msg)
        print()
        messages += [{"role": "assistant", "content": "".join(chat)}]
        logger.debug(f"Chat completion finished: {messages}")
        return messages


def fallback_model(model: str) -> str:
    try:
        openai.Model.retrieve(model)
        return model
    except openai.InvalidRequestError:
        print(
            f"Model {model} not available for provided API key. Reverting "
            "to gpt-3.5-turbo. Sign up for the GPT-4 wait list here: "
            "https://openai.com/waitlist/gpt-4-api\n"
        )
        return "gpt-3.5-turbo"
