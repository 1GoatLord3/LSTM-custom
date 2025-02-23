# LSTM-custom
Financial Price Prediction Crypto - stocks - funds - Forex

   [Input] → [LSTM Layer] → [Dense Layer] → [Output]

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

# Technical Indicators Documentation

This repository provides implementations and explanations for key technical indicators used in trading:  
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

## 📌 Usage  
- **RSI**: Identify overbought/oversold conditions  
- **EMA50 & EMA200**: Trend-following indicators (Golden Cross & Death Cross)  
- **ATR**: Measures absolute volatility  
- **nATR**: Compares volatility relative to price  
