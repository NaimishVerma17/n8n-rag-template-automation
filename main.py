from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import re
from pathlib import Path

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

SYSTEM_PROMPT = """
I want you to create a **fully functional n8n workflow JSON**—no images or screenshots, only valid JSON. Follow these instructions precisely:

1. **Reference All Knowledge Base Files**  
   - Use the best practices and JSON structures from:
     - **n8n Tips & Tricks**  
     - **n8n Cheat Sheet Guide**  
     - **AI Agent Chatbot + LONG TERM Memory + Note Storage + Telegram** workflow  
     - Other sample workflows I've provided you
   - Make sure you incorporate the guidelines about node structure, connections, memory usage, and error handling.

2. **Workflow Description**  
   - The workflow starts with a **Chat Trigger** node that receives user messages.  
   - It flows into an **AI Agent** node, which should call multiple tools (e.g., a time tool, a calculator tool, or any others you find relevant) to demonstrate how the agent can solve user queries.  
   - Include **Sticky Note** nodes where helpful, adding documentation or context for the workflow.  
   - Ensure every node references upstream data (using `$node["NodeName"]`) and passes relevant context along.  

3. **Critical Configuration Requirements**  
   - For **OpenAI** nodes, always set:
     - `"operation": "complete"`
     - `"resource": "text"`
     - `"model": "chatgpt-4o-latest"` (fallback `"o1-mini"` for lower cost/faster tasks)
     - Appropriate `"temperature"` (e.g., 0.1 for precise tasks, 0.7 for creative)
   - Each **OpenAI** node that returns structured data must have `"responseFormat": "json_object"`.
   - Ensure **all nodes** are connected properly in `"connections"`.
   - Any code nodes must include error handling (try/catch).
   - Provide credential references (but do not expose actual keys).
   - Add final nodes or steps that clearly mark the workflow as complete.

4. **No Placeholders or Partial JSON**  
   - Do not provide placeholders like `"API_KEY_HERE"` or `[YOUR DOC ID]`. Use a generic reference if needed (e.g., `"{{ myCredentials }}"`).
   - Output the **entire** workflow in a code block. It should be copy-paste-ready, with minimal manual edits.

5. **Ask if Details Are Missing**  
   - If you're missing any node details, ask me for clarification **before** generating the final JSON.  

6. **Output Format**  
   - **Only** produce valid JSON in a code block (```json ... ```).
   - No images or screenshot attempts.  
   - No extraneous commentary outside the code block.  

**Your goal**: Provide a single, self-contained JSON file that can be pasted into n8n, representing a chat-to-AI-agent flow with relevant tools and sticky notes, referencing the knowledge base for best practices. If any node is unclear or not in your knowledge base, list those nodes first and request guidance. Otherwise, produce the final JSON as requested.
Here is the Context: 
{context}

Here is the my Question: 
{question}
"""

# ---- CONFIG ---- #
DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
OPENAI_API_KEY = ""
CLAUDE_API_KEY = ""
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
)
# ---- SETUP ---- #
app = FastAPI()

# Optional CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Load & Split Documents ---- #
docs = []


def load_documents_from_folder(folder_path):
    print("Load docs")
    supported_extensions = [".pdf", ".txt", ".md", ".docx"]
    text_extensions = []

    for file_path in Path(folder_path).rglob("*"):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            try:
                if suffix in supported_extensions:
                    loader = UnstructuredFileLoader(str(file_path))
                    file_docs = loader.load()
                elif suffix in text_extensions:
                    content = file_path.read_text(encoding="utf-8")
                    file_docs = [
                        Document(
                            page_content=content, metadata={"source": str(file_path)}
                        )
                    ]
                else:
                    continue  # skip unsupported
                docs.extend(file_docs)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    print("Total docs: ", len(docs))
    return docs


def load_documents():
    raw_docs = load_documents_from_folder(DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(raw_docs)
    return docs


# ---- Initialize Vector DB ---- #
def init_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        print("Init VS")
        docs = load_documents()
        vectorstore = Chroma.from_documents(
            docs, embeddings, persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


# ---- Chat Chain Setup ---- #
def get_conversational_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
    # human_message_prompt = HumanMessagePromptTemplate.from_template(
    #     """ 
    #     Context:
    #     {context}
    #     Question:
    #         {question}
    #     """
    # )
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    callback_manager = CallbackManager(
        [StreamingStdOutCallbackHandler()]
    )  # 👈 for streaming to console

    # Anthropic model
    
    llm = ChatAnthropic(
        anthropic_api_key=CLAUDE_API_KEY,
        model="claude-3-opus-20240229",
        temperature=0,
        streaming=True,
        callback_manager=callback_manager,
    )

    # Open AI model

    # llm = ChatOpenAI(
    #     openai_api_key=OPENAI_API_KEY,
    #     model_name="gpt-4o",  # or "gpt-4" or "gpt-3.5-turbo"
    #     temperature=0,
    #     streaming=True,
    #     callback_manager=callback_manager,
    # )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return qa_chain, memory, prompt


# ---- Init everything on startup ---- #
vectorstore = init_vectorstore()
qa_chain, memory, debug_prompt_template  = get_conversational_chain(vectorstore)


# ---- Request Model ---- #
class Query(BaseModel):
    question: str


def extract_json_block(text: str):
    # Find content between ```json and ```
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in the text.")

    json_str = match.group(1).strip()

    try:
        # Parse JSON
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


@app.get("/")
def health():
    return {"success": True}


@app.get("/chat")
def chat(question):
    try:
        if not vectorstore or not qa_chain:
            {"error": "Vector store not initialised!"}
        result = qa_chain({"question": question})
        print(result)
        return {"response": result["answer"]}
        # return extract_json_block(result['result'])
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/debug")
def debug_prompt(question: str):
    try:
        # Step 1: Retrieve documents
        retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(question)
        
        # Step 2: Format context and history
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        chat_history = memory.chat_memory.messages
        
        # Step 3: Build inputs and format prompt
        _inputs = {
            "question": question,
            "chat_history": chat_history,
            "context": context_text
        }
        rendered_prompt = debug_prompt_template.format(**_inputs)
        
        print(rendered_prompt)

        # Convert prompt parts to readable text
        # formatted_parts = []
        # for msg in rendered_prompt:
        #     role = msg.type  # 'system', 'human', etc.
        #     content = msg.content
        #     formatted_parts.append(f"{role.upper()}:\n{content}\n")

        return {"prompt": rendered_prompt}

    except Exception as e:
        return {"error": str(e)}
