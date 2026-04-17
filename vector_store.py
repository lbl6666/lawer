import os
import config
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def load_all_txt_files(folder_path):
    """使用 TextLoader 加载文件夹下所有 txt 文件，返回 Document 列表（每个文件一个 Document）"""
    docs = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path, encoding='utf-8')
        # loader.load() 返回一个列表，通常只有一个 Document
        file_docs = loader.load()
        # 可以添加额外元数据，如文件名（loader 已自动添加 source）
        docs.extend(file_docs)
    return docs

def split_by_line(docs):
    """
    将每个文件的 Document 按换行符分割，每行成为一个新的 Document，
    并继承原文件的元数据（添加行号）
    """
    split_docs = []
    for doc in docs:
        lines = doc.page_content.splitlines()
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            metadata = doc.metadata.copy()
            metadata["line"] = i
            split_docs.append(Document(page_content=line, metadata=metadata))
    return split_docs


if __name__ == "__main__":
    # 配置
    txt_folder = "./subdata"          # 存放所有 txt 文件的文件夹       
    embeddings = OllamaEmbeddings(model=config.embeddings_model)

    # 1. 加载文档
    print("正在加载法律条文...")
    docs = load_all_txt_files(txt_folder)
    print(f"共加载 {len(docs)} 条原始条文")

    line_docs = split_by_line(docs)
    print(f"按行分割后共有 {len(line_docs)} 条法律条文")
    
    vectorstore = Chroma.from_documents(
        documents=line_docs,
        embedding=embeddings,
        persist_directory=config.persist_dir
    )
    
    print("全部完成！")