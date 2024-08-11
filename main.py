from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory,
)
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("chat_history.json"), # Temporarily commenting out as this doesn't play well with ConversationSummaryMemory
    memory_key="conversation_messages",
    return_messages=True,  # Return messages with Human/AI object context (rather than just a regular string)
    llm=chat,
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

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
