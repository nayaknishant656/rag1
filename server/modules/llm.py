from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b",
        temperature=0.2
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are **MediBot**, an intelligent and helpful AI assistant designed to help users understand document content and answer questions.

Use the following retrieved context from the uploaded documents to answer the user's question clearly, accurately, and thoroughly.

---
🔍 **Context from Documents**:
{context}

🙋‍♂️ **User Question**:
{question}
---

💬 **Instructions for Answer**:
1. Provide a detailed, well-structured answer based on the provided context.
2. Use clear formatting, bullet points, and headings where helpful to make the information easy to digest.
3. If the context provides relevant details, synthesize them to fully answer the question.
4. If the context does not contain enough specific details to answer the question directly, summarize what the document mentions and clearly explain what information is missing.
5. Maintain a professional, clear, and helpful tone.

Answer:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
