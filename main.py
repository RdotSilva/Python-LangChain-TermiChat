from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("chat_history.json"),
    memory_key="conversation_messages",
    return_messages=True,  # Return messages with Human/AI object context (rather than just a regular string)
)

prompt = ChatPromptTemplate(
    input_variables=["content", "conversation_messages"],
    messages=[
        MessagesPlaceholder(
            variable_name="conversation_messages"
        ),  # Specifically look for the conversation_messages memory_key defined above in the memory
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
