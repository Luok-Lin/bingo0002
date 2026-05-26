import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

try:
    from dl.kronos_predictor import KronosAdapter
except Exception:
    KronosAdapter = None

class StockTrendPredictor(nn.Module):
    """
    简单的 LSTM 模型用于预测带有连续因子的时间序列趋势
    """
    def __init__(self, input_size=10, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        predictions = self.linear(last_out)
        # 返回 batch 输出 (N, output_size)
        return predictions.squeeze(-1)

class DLEngine:
    def __init__(self, weight_path="models/weights/pretrained_model.pth"):
        self.backend = str(os.getenv("DL_BACKEND", "lstm")).strip().lower()
        self.kronos = None
        if self.backend in {"kronos", "auto"} and KronosAdapter is not None:
            self.kronos = KronosAdapter()
            if not self.kronos.available:
                print(f"[Kronos Adapter][WARN] 不可用，将回退到 LSTM: {self.kronos.unavailable_reason}")
        elif self.backend == "kronos":
            print("[Kronos Adapter][WARN] dl.kronos_predictor 导入失败，将回退到 LSTM。")

        self.model = StockTrendPredictor(input_size=10, hidden_layer_size=64, output_size=1)
        
        # 解析绝对路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.weight_path = os.path.join(base_dir, weight_path) if not os.path.isabs(weight_path) else weight_path
        self.scaler_path = f"{self.weight_path}.scaler.pkl"
        
        self.scaler = StandardScaler()
        self._load_or_init_weights()
        self._load_scaler()

    def _load_or_init_weights(self):
        os.makedirs(os.path.dirname(self.weight_path), exist_ok=True)
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path, map_location="cpu"))
            print(f"[DL Engine] 深度学习预测模型权重 [{self.weight_path}] 加载成功.")
        else:
            print("[DL Engine] 未发现预训练权重，随机初始化...")
            torch.save(self.model.state_dict(), self.weight_path)

    def _load_scaler(self):
        if not os.path.exists(self.scaler_path):
            return
        try:
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"[DL Engine] 标准化器 [{self.scaler_path}] 加载成功.")
        except Exception as exc:
            print(f"[DL Engine] 标准化器加载失败，将在预测时临时拟合: {exc}")

    def _save_scaler(self):
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def train_on_history(self, df_hist: pd.DataFrame, window_size=10, epochs=20, lr=0.005):
        """真实使用前置历史数据训练网络"""
        print("[DL Engine] 启动真实数据环境训练...")

        # 使用可用设备（优先 GPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"[DL Engine] 检测到 CUDA，可在 GPU 上训练: {torch.cuda.get_device_name(0)}")

        self.model.to(device)
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        if len(df_hist) <= window_size:
            print("[DL Engine] 数据不足，跳过训练。")
            return

        # 准备数据集
        raw_data = df_hist[cols].values
        self.scaler.fit(raw_data)
        scaled_data = self.scaler.transform(raw_data)

        X, y = [], []
        for i in range(len(scaled_data) - window_size - 1):
            X.append(scaled_data[i : i + window_size])
            # 预测第二天涨跌幅作为标签
            target_pnl = df_hist.iloc[i + window_size + 1]['涨跌幅'] / 10.0 # 缩放标签以便收敛
            y.append(target_pnl)

        if not X:
            return

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.FloatTensor(np.array(y))

        # 将数据移动到设备并创建 DataLoader 批量训练
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = min(32, max(1, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        X_tensor = X_tensor.to(device)
        y_tensor = y_tensor.to(device)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                preds = self.model(xb)
                # preds shape: (B,) or (B,1)
                if preds.dim() == 2 and preds.size(1) == 1:
                    preds = preds.squeeze(1)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            avg_loss = epoch_loss / len(dataset)
            if (epoch+1) % max(1, epochs//5) == 0 or epoch == epochs-1:
                print(f"  -> Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

        # 持久化到 CPU 兼容的状态字典
        self.model.to('cpu')
        torch.save(self.model.state_dict(), self.weight_path)
        self._save_scaler()
        print(f"[DL Engine] 模型使用真实数据微调完毕，已保存至 {self.weight_path}")

    def _extract_feature_matrix(self, features) -> np.ndarray:
        if isinstance(features, dict):
            features = features.get("dl_features", features.get("features", features))
        return np.asarray(features, dtype=float)

    def predict(self, ticker: str, features) -> dict:
        if isinstance(features, dict) and self.kronos is not None and self.kronos.available:
            kronos_ohlcv = features.get("kronos_ohlcv")
            if kronos_ohlcv is None:
                kronos_ohlcv = features.get("ohlcv")
            if kronos_ohlcv is not None:
                try:
                    return self.kronos.predict(ticker, kronos_ohlcv)
                except Exception as exc:
                    if self.backend == "kronos":
                        print(f"[Kronos Adapter][WARN] 推理失败，将回退到 LSTM: {exc}")
                    else:
                        print(f"[Kronos Adapter][WARN] auto 后端推理失败，将回退到 LSTM: {exc}")

        features = self._extract_feature_matrix(features)
        self.model.eval()
        with torch.no_grad():
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(features[-10:])
            scaled_features = self.scaler.transform(features[-10:])
            seq = torch.FloatTensor(scaled_features).unsqueeze(0)
            raw_score = self.model(seq).item()

        confidence = min(np.abs(raw_score) * 100, 99.9)
        trend = "上涨 (看多)" if raw_score > 0 else "下跌 (看空)"
        return {
            "score": round(raw_score * 10, 4), # 放大因为预测时压了一倍
            "trend": trend,
            "confidence": f"{round(confidence, 2)}%"
        }
