
![logo](https://github.com/user-attachments/assets/2de19ff0-9c2b-453f-85e5-00a8c57b1252)

# 📈 LSTM-Custom  
### Financial Price Prediction for Crypto, Stocks, Funds, and Forex  

This repository implements **LSTM-based financial market prediction models** combined with essential **technical indicators** to enhance trading strategies.  

## 🧠 LSTM Neural Network  

LSTM (Long Short-Term Memory) is a type of **Recurrent Neural Network (RNN)** designed to remember long-term dependencies in time series data, making it ideal for **financial forecasting**.  

### 🔹 Basic LSTM Structure  

  ┌───────────────────┐
  │  Forget Gate (f)  │  🔥 Forget irrelevant past info
  ├───────────────────┤
  │   Input Gate (i)  │  📥 Learn new relevant data
  ├───────────────────┤
  │ Output Gate (o)   │  🎯 Decide what to output
  ├───────────────────┤
  │ Cell State (c)    │  🧠 Long-term memory
  └───────────────────┘

#🏗️ ML Frameworks: Scikit-Learn vs. Keras
  🛠 Scikit-Learn
  Great for classic machine learning (Random Forest, SVM, etc.)
  Preprocessing & feature selection tools
  Simple, efficient, but not ideal for deep learning

  🔥 Keras (with TensorFlow backend)
  High-level API for deep learning models
  Supports LSTM, CNN, and transformers
  Easier to prototype than raw TensorFlow


---

## 🏗️ ML Frameworks: Scikit-Learn vs. Keras  

### 🛠 **Scikit-Learn**  
✅ Best for **classic machine learning** (Random Forest, SVM, Decision Trees)  
✅ Powerful **preprocessing & feature selection** tools  
✅ **Simple and efficient**, but not ideal for deep learning  

### 🔥 **Keras (with TensorFlow backend)**  
✅ **High-level API** for deep learning models  
✅ Supports **LSTM, CNN, and transformers**  
✅ **Easy to prototype** compared to raw TensorFlow  

---

# 📊 Technical Indicators Documentation  

This repository provides implementations and explanations for key **technical indicators** used in financial trading:  
**RSI (Relative Strength Index), EMA50 (Exponential Moving Average 50), EMA200, ATR (Average True Range), and nATR (Normalized ATR).**  

## 🛠️ Technical Indicators  

### 1️⃣ Relative Strength Index (RSI)  
The **RSI (Relative Strength Index)** measures the magnitude of recent price changes to evaluate overbought or oversold conditions.  

**Formula:**  
$$
RSI = 100 - \left( \frac{100}{1 + RS} \right)
$$
where:  
$$
RS = \frac{\text{Average Gain over } n \text{ periods}}{\text{Average Loss over } n \text{ periods}}
$$
- **Overbought**: RSI > 70  
- **Oversold**: RSI < 30  
- Default period: **14**  

---

### 2️⃣ Exponential Moving Average (EMA)  
The **Exponential Moving Average (EMA)** gives more weight to recent prices, making it more responsive than a Simple Moving Average (SMA).  

**Formula:**  
$$
EMA_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA_{t-1}
$$
where:  
$$
\alpha = \frac{2}{n+1}
$$
- **EMA50**: \( n = 50 \)  
- **EMA200**: \( n = 200 \)  

---

### 3️⃣ Average True Range (ATR)  
The **Average True Range (ATR)** is a volatility indicator that measures market fluctuations.  

**Formula:**  
$$
ATR_n = \frac{1}{n} \sum_{i=1}^{n} TR_i
$$
where **True Range (TR)** is:  
$$
TR = \max( H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}| )
$$
- Default period: **14**  
- Higher ATR → **Higher volatility**  
- Lower ATR → **Lower volatility**  

---

### 4️⃣ Normalized ATR (nATR)  
The **Normalized ATR (nATR)** standardizes ATR against price, making it more comparable across different assets.  

**Formula:**  
$$
nATR = \frac{ATR}{P} \times 100
$$
where:  
- \( ATR \) = Average True Range  
- \( P \) = Current Price  

For example, if ATR = $5$ and price = <span>$</span>100, then:  
$$
nATR = \frac{5}{100} \times 100 = 5\%
$$  

---

## 📌 How to Use These Indicators  

🔹 **RSI**: Identify overbought/oversold conditions for possible reversals  
🔹 **EMA50 & EMA200**: Trend-following indicators (Golden Cross & Death Cross)  
🔹 **ATR**: Measures absolute volatility to set stop-loss levels  
🔹 **nATR**: Compares volatility relative to price for better risk assessment  

