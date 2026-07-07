import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
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

UPLAOD_DIR = "./uploaded_files"
os.makedirs(UPLAOD_DIR, exist_ok=True)


GLOBAL_RETRIEVER = None

@app.post("/api/upload", summary="上傳 PDF 文件")
def upload_pdf(file: UploadFile = File(...)):
    global GLOBAL_RETRIEVER

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支援上傳 PDF 文件")
    
    try:
        file_path = os.path.join(UPLAOD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"【上傳成功】檔案已儲存至: {file_path}，開始進行向量化")

        loader = PyPDFLoader(file_path)
        docs =loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        GLOBAL_RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 3})

        return {
            "message": f"文件 {file.filename} 上傳成功，開始進行向量化",
            "file_name": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理檔案發生錯誤:{str(e)}")

class QueryRequest(BaseModel):
    question: str

class SourceCitation(BaseModel):
    page: int
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]



@app.post("/api/query", response_model=QueryResponse, summary="針對已上傳文件進行RAG問答")
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