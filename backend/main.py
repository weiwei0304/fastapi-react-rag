import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="DocuMind AI RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

GLOBAL_RETRIEVER = None
@app.on_event("startup")
def initialize_backend_rag():
    global GLOBAL_RETRIEVER
    pdf_file = "test.pdf"

    if os.path.exists(pdf_file):
        print("[系統初始化]正在預先讀取並建立 test.pdf 的向量庫...")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("[系統初始化]完成 test.pdf 的向量庫建立!")
    else:
        print("[系統初始化]錯誤: test.pdf 檔案不存在")
        raise FileNotFoundError(f"文件 {pdf_file} 不存在")

class QueryRequest(BaseModel):
    question: str

class SourceCitation(BaseModel):
    page: int
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]



@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Google API Key 未設定")
    if GLOBAL_RETRIEVER is None:
        raise HTTPException(status_code=500, detail="RAG 系統尚未初始化")

    try:
        user_question = request.question
        
        retrieved_docs = GLOBAL_RETRIEVER.invoke(user_question)

        context_text = ""
        sources_list=[]

        for i, doc in enumerate(retrieved_docs):
            page_num = doc.metadata.get("page", 0) + 1
            context_text += f"\n[文件第 {page_num} 頁段落]:\n{doc.page_content}\n"

            sources_list.append(
                SourceCitation(page=page_num, content=doc.page_content.strip())
            )
        
        full_prompt = (
            "你是一個專業的文件分析助手。請嚴格根據下方提供的【背景知識】來回答使用者的問題。\n"
            "如果你在背景知識中找不到答案，請誠實回答『我在文件中找不到相關資訊』，絕對不要胡編亂造。\n\n"
            f"【背景知識】:\n{context_text}\n"
            f"【問題】: {user_question}"
        )

        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt
        )

        return QueryResponse(answer=response.text, sources=sources_list)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器錯誤: {str(e)}")