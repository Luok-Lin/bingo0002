import json
import os
import datetime
from .db_middleware import DatabaseMiddleware
try:
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError:
    pass

class MemoryBank:
    """长期记忆与知识库，包含关系型数据库双写与向量RAG检索引擎"""
    
    # 核心超参数配置：将业务阈值等数值化参数与代码逻辑隔离
    CONFIG = {
        "INIT_FITNESS": 1.0,             # GEP: 默认基因适应度基准
        "REWARD_FITNESS": 0.1,           # GEP: 成功复验单次适应度提升
        "PENALTY_FITNESS": 0.2,          # GEP: 亏损复验单次适应度衰减
        "FORGET_THRESHOLD": 0.2,         # GEP: 物理遗忘分值下限 (淘汰基因)
        "RETRIEVE_MIN_SCORE": 0.3,       # 允许被RAG检索的最底分
        "CRYSTALLIZE_WIN_THRESHOLD": 1.3,# 高质量致胜经验结晶门槛 (突变触发点)
        "CRYSTALLIZE_LOSS_THRESHOLD": 0.8,# 低分教训证伪素材门槛
        "CRYSTALLIZE_MIN_COUNT": 3,      # 至少需要几条高分经验触发结晶
        "CRYSTALLIZE_MAX_ANTI_EXAMPLES": 5 # 最多附带多少条近期反例用于证伪
    }

    def __init__(self, file_path="data/json/reflections.json", db_path="data/db/tradingagents.db", 
                 persist_directory="data/vector_db/advanced_reflections_db", 
                 principle_file="data/json/principles.json"):
        # 获取项目根目录 (相对于 memory_bank.py 的父级的绝对路径)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.file_path = os.path.join(base_dir, file_path) if not os.path.isabs(file_path) else file_path
        self.db = DatabaseMiddleware(os.path.join(base_dir, db_path) if not os.path.isabs(db_path) else db_path)
        self.memory = self.load()
        
        self.persist_directory = os.path.join(base_dir, persist_directory) if not os.path.isabs(persist_directory) else persist_directory
        self.principle_file = os.path.join(base_dir, principle_file) if not os.path.isabs(principle_file) else principle_file
        self.principles = self._load_principles()
        
        # 向量知识库初始化 (高维空间存储)
        print("[MemoryBank] 正在初始化 HuggingFace Embeddings 与 ChromaDB 以支持记忆 RAG...")
        # 强制使得 langchain/huggingface 不去走被封的大陆直连或者受到无代理影响
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vector_store = Chroma(
                collection_name="agent_experiences",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            print(f"[MemoryBank] 向量知识库初始化失败: {e}")
            self.vector_store = None

    def _load_principles(self):
        if os.path.exists(self.principle_file):
            try:
                with open(self.principle_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_principles(self):
        with open(self.principle_file, "w", encoding="utf-8") as f:
            json.dump(self.principles, f, ensure_ascii=False, indent=4)

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def append(self, record):
        """兼容老接口录入基础经验日志，并自动同步存入向量数据库"""
        self.memory.append(record)
        # 双写：保持JSON作为日志，但写入真实的SQLite供核心查询
        self.db.insert_reflection(record)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)
            
        # 触发 向量化存储(RAG)
        self.append_experience(
            ticker=record.get("ticker", "UNKNOWN"),
            role="System", 
            sentiment_or_regime="General", # 可以通过后面的更新扩充
            content=record.get("reflection_text", ""), 
            action_taken=record.get("decision", "HOLD")
        )
        
    def append_experience(self, ticker: str, role: str, sentiment_or_regime: str, content: str, action_taken: str, pnl_percent: float = 0.0):
        """高级特性1&2：记录经验并打上角色、市场环境标签"""
        if not self.vector_store or not content:
            return None
            
        doc_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{ticker}_{role}"
        
        # 将最新的 PnL 转化为初步的 score：赚钱的话增加其初始权重
        base_score = self.CONFIG.get("INIT_SCORE", 1.0)
        initial_score = base_score + (pnl_percent / 10.0) if pnl_percent > 0 else base_score
        
        metadata = {
            "date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "ticker": ticker,
            "role": role,                               # 方向5: 分角色独立记忆
            "market_regime": sentiment_or_regime,       # 方向2: 区分市场环境
            "action_taken": action_taken,
            "score": initial_score,                     # 方向4: 置信度分数打分反馈基准
            "crystallized": False                       # 方向3: 是否已结晶为公理
        }
        
        doc = Document(page_content=content, metadata=metadata)
        self.vector_store.add_documents([doc], ids=[doc_id])
        return doc_id

    def retrieve_relevant_experience(self, current_scene_desc: str, role: str, current_regime: str, top_k=3):
        """高级特性1：记忆RAG化，检索最相近历史教训"""
        if not self.vector_store:
            return []
            
        filter_dict = {
            "$and": [
                {"role": {"$eq": role}},
                # 大量数据积攒后可以开启 {"market_regime": {"$eq": current_regime}} 的强过滤
                {"score": {"$gte": self.CONFIG.get("RETRIEVE_MIN_SCORE", 0.3)}}               # 屏蔽掉被扣分验证无效的垃圾经验
            ]
        }
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=current_scene_desc, 
                k=top_k, 
                filter=filter_dict
            )
            return [{"content": res[0].page_content, "score": res[0].metadata.get("score"), "dist": res[1]} for res in results]
        except Exception:
            return []

    def update_experience_score_by_action(self, ticker: str, action_taken: str, pnl_result: float):
        """高级特性4：强化学习雏形，根据最新策略收益修改同类历史经验的权威性"""
        if not self.vector_store: return
        
        try:
            # 找到过去做出类似决定的向量文档 (简化操作，找出该 ticker 最近1个月的同向决策)
            # 在 ChromaDB 里比较复杂，所以我们采用 get 全部比对过滤的方式
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs['metadatas']: return
            
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('ticker') == ticker and metadata.get('action_taken') == action_taken:
                    doc_id = all_docs['ids'][i]
                    current_score = metadata.get("score", self.CONFIG.get("INIT_SCORE", 1.0))
                    
                    if pnl_result > 0:
                        metadata["score"] = current_score + self.CONFIG.get("REWARD_SCORE", 0.1) # 奖赏
                    elif pnl_result < 0:
                        metadata["score"] = current_score - self.CONFIG.get("PENALTY_SCORE", 0.2) # 严厉惩罚
                        
                    content = all_docs['documents'][i]
                    new_doc = Document(page_content=content, metadata=metadata)
                    
                    if metadata["score"] < self.CONFIG.get("FORGET_THRESHOLD", 0.2):
                        self.vector_store.delete(ids=[doc_id])  # 物理遗忘
                        print(f"🔄 经验[{doc_id}]连续亏损扣分归零，已被物理遗忘。")
                    else:
                        self.vector_store.delete(ids=[doc_id])
                        self.vector_store.add_documents([new_doc], ids=[doc_id])
        except Exception as e:
            pass

    def crystallize_knowledge(self, llm_scorer):
        """高级特性3：定期结晶，提取核心交易法则"""
        if not self.vector_store: return None
        all_docs = self.vector_store.get()
        if not all_docs or not all_docs['metadatas']: return None
        
        high_score_docs = []
        low_score_docs = [] # 记录低分/亏损经验用作证伪素材
        doc_ids_to_mark = []
        
        for i, m in enumerate(all_docs['metadatas']):
            score = m.get('score', self.CONFIG.get("INIT_SCORE", 1.0))
            if score > self.CONFIG.get("CRYSTALLIZE_WIN_THRESHOLD", 1.3) and not m.get('crystallized', False):
                high_score_docs.append(all_docs['documents'][i])
                if 'ids' in all_docs and all_docs['ids']:
                    doc_ids_to_mark.append(all_docs['ids'][i])
            elif score <= self.CONFIG.get("CRYSTALLIZE_LOSS_THRESHOLD", 0.8):
                low_score_docs.append(all_docs['documents'][i])
                
        if len(high_score_docs) < self.CONFIG.get("CRYSTALLIZE_MIN_COUNT", 3):    # 简化：只要有3条高质量就触发结晶
            return None
            
        print(f"💎 正在结合 {len(high_score_docs)} 条高价值经验与 {len(low_score_docs)} 条亏损教训进行证伪提炼...")
        # 提取过之后，马上将这些高分经验打上结晶标签（crystallized=True），避免系统在未来的每次循环中做无脑复读生成
        try:
            from langchain.schema import Document
            for i, doc_id in enumerate(doc_ids_to_mark):
                # 重新写入带crystallized标签的文档
                idx = all_docs['ids'].index(doc_id)
                old_content = all_docs['documents'][idx]
                old_metadata = all_docs['metadatas'][idx].copy()
                old_metadata['crystallized'] = True
                new_doc = Document(page_content=old_content, metadata=old_metadata)
                
                self.vector_store.delete(ids=[doc_id])
                self.vector_store.add_documents([new_doc], ids=[doc_id])
            print("🔒 已成功将本次提取的经验标记为【已结晶】，防止重复反思。")
        except Exception as e:
            pass

        # 构建同时具有正面和反面教材的推理上下文
        combined_text = "【当前的高价值致胜经验】:\n" + "\n".join(high_score_docs)
        if low_score_docs:
            max_anti = self.CONFIG.get("CRYSTALLIZE_MAX_ANTI_EXAMPLES", 5)
            combined_text += "\n\n【近期低分或亏损的失败教训(作为证伪反例)】:\n" + "\n".join(low_score_docs[-max_anti:]) # 取近期最惨痛教训
            
        # return combined_text 给更高层的 Agent 进行大模型处理
        return combined_text
        
    def get_recent_reflections(self, k=3, ticker=None):
        if ticker:
            # 升级：从强大的数据库中检索以模拟原版健壮的存储引擎
            db_records = self.db.get_reflections(ticker, limit=k)
            # 为了防止初次运行时数据库没数据但内存有：
            if db_records:
                # 把列表反转一下因为DB是DESC取出的
                return db_records[::-1]
            else:
                filtered_memory = [m for m in self.memory if m.get("ticker") == ticker]
                return filtered_memory[-k:]
        return self.memory[-k:]