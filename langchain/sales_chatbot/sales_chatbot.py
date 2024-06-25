from typing import List
import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from pathlib import Path


current_file_path = Path(__file__).absolute()
# 构建相对于当前文件的默认数据文件路径
default_data_path = current_file_path.parent / 'real_estate_sales_data.txt'
openai_baseurl, openai_api_key = os.getenv('OPENAI_URL'), os.getenv('OPENAI_API_KEY')

    
# 接收文件并加载到faiss local index中
def process_file(file_path, category='default') -> List[Document]:
    # 使用category 对文档进行标注归类
    text_splitter = CharacterTextSplitter(        
        separator = "\n\n",
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = False,
    )
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            text = f.read()
            # 将text base64编码生成唯一hash
            doc_hash = hash(text)
            docs = text_splitter.create_documents([text])
            # 增加分类元数据
            for doc in docs:
                doc.metadata["category"] = category
                doc.metadata['name'] = f.name
                doc.metadata['id'] = doc_hash
            return docs


class ChatBot:
    def __init__(self, default_data_path, base_url, api_key):
        self.default_data_path = default_data_path
        self.enable_chat = True
        self.faiss_index = 'my-bot'
        self.embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, base_url=base_url, api_key=api_key)
        self.category = 'realEstate'
        self.vector_db: FAISS = self.initialize_vector_db()
        self.BOT = self.set_bot()

    def initialize_vector_db(self):
        faiss_db = None
        if self.default_data_path and os.path.exists(self.default_data_path):
            docs = process_file(self.default_data_path, category='realEstate')
            # 判断本地索引 self.faiss_index 是否存在
            if os.path.exists(self.faiss_index):
                faiss_db = FAISS.load_local(self.faiss_index, self.embeddings, allow_dangerous_deserialization=True)
                faiss_db.add_documents(docs)
            else:
                faiss_db = FAISS.from_documents(docs, self.embeddings)
                faiss_db.save_local(self.faiss_index)
        return faiss_db
    
    def set_bot(self, category='realEstate'):
        print(f"检索主题切换为: {category}")
        retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.5,'filter': {'category': category}})
        
        bot = RetrievalQA.from_chain_type(self.llm, retriever=retriever)
        # 返回向量数据库的检索结果
        bot.return_source_documents = True

        return bot

    def update_db(self, file_path: str, category: str):
        docs = process_file(file_path, category)
        self.vector_db.add_documents(docs)
        self.vector_db.save_local(self.faiss_index)
    
    def process_and_reload(self, file, selected_topic):
        if file:
            file_path = file.name
            self.update_db(file_path, selected_topic)
            self.BOT = self.set_bot(category=selected_topic)
            return "文件已成功处理并更新至数据库。"

    def chat_wrapper(self, message, history, selected_topic):
        print(f"[message]{message}")
        print(f"[history]{history}")
        self.BOT = self.set_bot(category=selected_topic)
        ans = self.BOT({"query": message})
        # 如果检索出结果，或者开了大模型聊天模式
        # 返回 RetrievalQA combine_documents_chain 整合的结果
        if ans["source_documents"] or self.enable_chat:
            print(f"[result]{ans['result']}")
            print(f"[source_documents]{ans['source_documents']}")
            # 将新消息添加到历史记录中，并构造正确的消息对格式返回
            new_history = history + [[message, ans["result"]]]
            return new_history
        # 否则输出套路话术
        else:
            # 同样构造消息对格式返回
            new_history = history + [[message, "这个问题我要问问领导"]]
            return new_history

    def on_topic_change(self, topic):
        print(f"主题已更改为: {topic}")
        self.BOT = self.set_bot(category=topic)


    def launch_gradio(self):
        topics = ["realEstate", "homeAppliances", "devOps"]
        with gr.Blocks() as demo:
            # 主题选择
            topic_selection = gr.Radio(label="选择主题", choices=topics, type="value")
            topic_selection.change(fn=self.on_topic_change, inputs=[topic_selection], outputs=None) 
            # 文件上传
            file_upload = gr.File(label="上传文件")
            file_upload_output = gr.Textbox(label="文件上传结果")
            file_upload.upload(fn=self.process_and_reload, inputs=[file_upload, topic_selection], outputs=file_upload_output)
            
            # 聊天界面
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="输入你的问题")
            msg.submit(self.chat_wrapper, [msg, chatbot, topic_selection], chatbot)
            msg.submit(None, None, chatbot, queue=False)  # 清空输入框
            
        # demo.launch()
        demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":

    app = ChatBot(
        default_data_path=str(default_data_path),
        base_url=openai_baseurl,
        api_key=openai_api_key,
    )
    app.launch_gradio()
