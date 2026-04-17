from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import config
from bailian_client import create_bailian_client, create_query_rewriter, create_reranker


class RagService(object):
    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位精通中国法律的专业人士。请依据以下检索到的法律条例，"
             "准确、专业地回答用户的问题。回答时应优先使用文档中的信息，"
             "并结合你的法律知识，但必须确保所有陈述都有据可依，"
             "尽可能引用具体的法律条文或来源。如果文档内容不足以回答问题，"
             '请明确告知"根据提供的材料无法回答该问题"，切勿编造信息。'
             "回答需清晰，符合法律专业表达习惯。"
             "\n\n检索到的法律条例如下:\n{context}。"),
            ("user", "{input}"),
        ])

        # ===== 原有组件（保持 Ollama 本地模型）=====
        self.chat_model = ChatOllama(model="lawer")
        self.embeddings = OllamaEmbeddings(model=config.embeddings_model)

        # ===== 百炼平台组件（查询改写 + 重排序）=====
        self.bailian_client = create_bailian_client()
        self.query_rewriter = create_query_rewriter(self.bailian_client)
        self.reranker = create_reranker(self.bailian_client, top_n=config.rerank_k)

        self.chain = self.__get_chain()

    def __get_chain(self):
        # ===== 向量存储 & 召回（使用原始 Embedding 模型）=====
        vectorstore = Chroma(
            persist_directory=config.persist_dir,
            embedding_function=self.embeddings,
        )
        # 召回阶段取更多候选文档，交由重排序筛选
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.retrieval_k}
        )

        # ===== 文档格式化 =====
        def format_document(docs: list[Document]) -> str:
            if not docs:
                return "无相关参考资料"
            formatted = ""
            for doc in docs:
                formatted += f"参考文档：{doc.page_content}\n"
            return formatted

        # ===== 增强检索函数：改写 → 召回 → 重排 =====
        def enhanced_retrieve(query: str) -> str:
            # Step 1：查询改写（使用百炼 Qwen3-4B）
            rewritten_query = self.query_rewriter.rewrite(query)
            print(f"[查询改写] 原始: {query}")
            print(f"[查询改写] 改写: {rewritten_query}")

            # Step 2：向量召回（使用改写后的查询）
            docs = retriever.invoke(rewritten_query)
            print(f"[向量召回] 召回文档数: {len(docs)}")

            # Step 3：重排序（使用百炼 Qwen3-rerank）
            reranked_docs = self.reranker.rerank(rewritten_query, docs)
            print(f"[重排序]   重排后文档数: {len(reranked_docs)}")

            # Step 4：格式化送生成
            return format_document(reranked_docs)

        # ===== RAG Chain =====
        chain = (
            {
                "input": RunnablePassthrough(),
                # context 由增强检索函数产生（改写+召回+重排）
                "context": RunnableLambda(lambda x: x["input"])
                | RunnableLambda(enhanced_retrieve),
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        return chain


if __name__ == "__main__":
    rag = RagService()
    question = "我和别人签了租赁合同，对方现在不付钱了，我还能不能要回这笔钱？"
    print(f"【用户问题】{question}\n")
    res = rag.chain.invoke({"input": question})
    print("\n【生成回答】")
    print(res)
