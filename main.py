import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 加入 CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stock-app-1-a1ft.onrender.com"],  # 或指定你的 Flutter App 的網址
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有方法，包括 OPTIONS
    allow_headers=["*"],  # 允許所有 headers
)

class StockInput(BaseModel):
    stoid: str

@app.post("/predict")
def predict(data: StockInput):

    stoid = data.stoid
    # 1️⃣ 下載 資料
    fn = 'ai_predict.csv'

    df = fmind.taiwan_stock_daily(stock_id = stoid, start_date = '2015-01-01')
    df = df.drop(['Trading_money', 'stock_id', 'spread', 'Trading_turnover'], axis=1)
    df.columns = ['Date', 'Volume', 'Open', 'High', 'Low', 'Close']
    df.set_index("Date" , inplace=True)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
    df = df.reset_index(drop=False)

    df['MA20'] = df['Close'].rolling(window=20).mean()

    mm = np.array(df['Close'].rolling(window=20).mean())
    pp = [0]
    for i in range(1,len(mm)):
      bb=(((mm[i]-mm[i-1])*100)/mm[i-1])
      pp.append(bb)
    df['MA20p'] = pp

    # OBV 計算
    obv = [0]
    clo = df['Close']

    for i in range(1, len(df)):
        if clo.iloc[i] > clo.iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif clo.iloc[i] < clo.iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    obv = pd.DataFrame(obv)
    obv = obv.ewm(span=12).mean()
    df['OBV'] = obv

    # 標籤：預測隔日漲跌
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # 2️⃣ 特徵與標籤
    features = ['MA20', 'MA20p', 'OBV']
    X = df[features].values
    y = df['Target'].values

    # 3️⃣ 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4️⃣ 建構模型
    class MAOBVModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = MAOBVModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5️⃣ 模型訓練
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    for epoch in range(100):
        model.train()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 6️⃣ 預測明日漲跌（使用最新一筆資料）
    latest_features = X_scaled[-1:]
    latest_tensor = torch.tensor(latest_features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
      score = model(latest_tensor).item()
      #direction = "📈 預測：明日可能上漲" if prediction > 0.5 else "📉 預測：明日可能下跌"
      #score = 0.7 * data.ma20 + 0.3 * data.obv
      return {"prediction": score}



