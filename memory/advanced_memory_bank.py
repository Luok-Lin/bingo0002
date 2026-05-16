import json
import os
import datetime
from typing import List, Dict, Optional
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None
from langchain_core.documents import Document

class AdvancedMemoryBank:
    """
    进化认知引擎 (Evolutionary Cognitive Engine) 
    支持特性：
    1. 经验RAG化 (Chroma 向量检索)
    2. 行情标签 (Market Regime 匹配)
    3. 经验遗忘与打分强化机制
    4. 经验结晶 (Principles)
    5. 角色专属记忆分离
    """
    def __init__(self, persist_directory="data/vector_db/advanced_reflections_db", 
                 principle_file="data/json/principles.json"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.persist_directory = os.path.join(base_dir, persist_directory) if not os.path.isabs(persist_directory) else persist_directory
        self.principle_file = os.path.join(base_dir, principle_file) if not os.path.isabs(principle_file) else principle_file
        
        # 初始化嵌入模型 (经验RAG化)
        if HuggingFaceEmbeddings is None:
            print("[AdvancedMemoryBank] WARNING: HuggingFaceEmbeddings 未能导入，若需向量化请安装: pip install -U langchain-huggingface")
        else:
            # 动态寻找本地模型路径
            local_model_base = os.path.join(base_dir, "data", "models", "all-MiniLM-L6-v2", "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots")
            model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
            
            model_kwargs = {}
            if os.path.exists(local_model_base):
                snapshots = [d for d in os.listdir(local_model_base) if os.path.isdir(os.path.join(local_model_base, d))]
                if snapshots:
                    model_name_or_path = os.path.join(local_model_base, snapshots[0])
                    model_kwargs = {'local_files_only': True}
                    print(f"[AdvancedMemoryBank] 使用本地模型: {model_name_or_path}")

            self.embeddings = HuggingFaceEmbeddings(model_name=model_name_or_path, model_kwargs=model_kwargs)
        self.vector_store = Chroma(
            collection_name="agent_experiences",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.principles = self._load_principles()
        
    def _load_principles(self):
        if os.path.exists(self.principle_file):
            with open(self.principle_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_principles(self):
        with open(self.principle_file, "w", encoding="utf-8") as f:
            json.dump(self.principles, f, ensure_ascii=False, indent=4)
            
    def append_experience(self, role: str, sentiment_or_regime: str, content: str, action_taken: str):
        """记录经验，并打上角色和市场标签 (解决方向2 和 方向5)"""
        doc_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{role}"
        
        metadata = {
            "date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "role": role,                               # 方向5: 分角色独立记忆
            "market_regime": sentiment_or_regime,       # 方向2: 区分市场环境
            "action_taken": action_taken,
            "score": 1.0,                               # 方向4: 初始验证分数为 1.0
            "crystallized": False                       # 方向3: 是否已结晶为公理
        }
        
        doc = Document(page_content=content, metadata=metadata)
        self.vector_store.add_documents([doc], ids=[doc_id])
        return doc_id

    def retrieve_relevant_experience(self, current_scene_desc: str, role: str, current_regime: str, top_k=3):
        """经验RAG化 (方向1)：检索当前最匹配的历史经验"""
        # 利用 metadata 对角色和市场情绪做强过滤，保证牛市不看熊市经验，技术不管基本面
        filter_dict = {
            "$and": [
                {"role": {"$eq": role}},
                {"market_regime": {"$eq": current_regime}},
                {"score": {"$gte": 0.5}}               # 屏蔽掉经过验证无效的经验（方向4）
            ]
        }
        
        # 为了兼容可能某些角色经验还很少的情况，我们加一个异常回退
        try:
            results = self.vector_store.similarity_search_with_score(
                query=current_scene_desc, 
                k=top_k, 
                filter=filter_dict
            )
            # 返回提取出的内容
            return [{"content": res[0].page_content, "score": res[0].metadata.get("score")} for res in results]
        except Exception:
            return []

    def update_experience_score(self, doc_id: str, pnl_result: float):
        """经验验证与打分反馈闭环 (方向4)"""
        # pnl_result 是收益率，赚钱加经验分，亏钱扣经验分
        try:
            doc_data = self.vector_store.get(ids=[doc_id])
            if doc_data and doc_data['metadatas']:
                metadata = doc_data['metadatas'][0]
                current_score = metadata.get("score", 1.0)
                
                # 强化学习雏形：赚钱加分，亏钱扣分
                if pnl_result > 0:
                    metadata["score"] = current_score + 0.2
                else:
                    metadata["score"] = current_score - 0.5
                    
                # 更新回向量库 
                # (ChromaDB 的 update 操作比较特殊，这里简化演示对元数据的覆盖)
                content = doc_data['documents'][0]
                new_doc = Document(page_content=content, metadata=metadata)
                
                # 若分数过低（<0.2），启动边缘遗忘机制 (即下架该经验)
                if metadata["score"] < 0.2:
                    self.vector_store.delete(ids=[doc_id])
                else:
                    # 先删后加相当于 Update
                    self.vector_store.delete(ids=[doc_id])
                    self.vector_store.add_documents([new_doc], ids=[doc_id])
        except Exception as e:
            pass

    def crystallize_knowledge(self, llm_summarizer):
        """定期结晶与遗忘机制 (方向3): 把高分区细碎经验提炼为底层原则"""
        # 获取所有评分 > 1.5 的优秀经验
        all_docs = self.vector_store.get()
        if not all_docs or not all_docs['metadatas']: return
        
        high_score_docs = []
        for i, m in enumerate(all_docs['metadatas']):
            if m.get('score', 0) > 1.5 and not m.get('crystallized', False):
                high_score_docs.append(all_docs['documents'][i])
                
        if len(high_score_docs) < 5:
            return # 经验还不够多，不急着结晶
            
        # 这里传入你的 LLM，将高分区经验做摘要
        prompt = f"请把以下散乱的交易经验总结为 1~2条 核心黄金交易法则：\n{high_score_docs}"
        principle = llm_summarizer.invoke(prompt)
        
        self.principles.append({
            "date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "principle": principle.content if hasattr(principle, 'content') else str(principle)
        })
        self._save_principles()
        
        # TODO: 可选操作 - 把结晶后的原版细碎经验删掉(物理遗忘)或者标记 crystallized = True
