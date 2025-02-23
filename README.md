# LSTM-custom
Financial Price Prediction Crypto - stocks - funds - Forex

# Technical Indicators Documentation

This repository provides implementations and explanations for key technical indicators used in trading:  
**RSI (Relative Strength Index), EMA50 (Exponential Moving Average 50), EMA200, ATR (Average True Range), and nATR (Normalized ATR).**  

## 📌 Indicators  

### 1️⃣ Relative Strength Index (RSI)  
**Formula:**  
$
RSI = 100 - \left( \frac{100}{1 + RS} \right)
$
where:  
$
RS = \frac{\text{Average Gain over } n \text{ periods}}{\text{Average Loss over } n \text{ periods}}
$
- Default period: **14**  
- RSI values range between **0 and 100**  
- **Overbought:** RSI > 70  
- **Oversold:** RSI < 30  

---

### 2️⃣ Exponential Moving Average (EMA)  
EMA smooths price data with a higher weight on recent prices.  

**Formula:**  
\[
EMA_t = \alpha \cdot P_t + (1 - \alpha) \cdot EMA_{t-1}
\]
where:  
\[
\alpha = \frac{2}{n+1}
\]
- **EMA50**: \( n = 50 \)  
- **EMA200**: \( n = 200 \)  

---

### 3️⃣ Average True Range (ATR)  
ATR measures market volatility.  

**Formula:**  
\[
ATR_n = \frac{1}{n} \sum_{i=1}^{n} TR_i
\]
where **True Range (TR)** is:  
\[
TR = \max( H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}| )
\]
- Default period: **14**  

---

### 4️⃣ Normalized ATR (nATR)  
nATR standardizes ATR against price to assess relative volatility.  

**Formula:**  
\[
nATR = \frac{ATR}{P} \times 100
\]
where:  
- \( ATR \) = Average True Range  
- \( P \) = Current Price  

---

## 📌 Usage  
- **RSI**: Identify overbought/oversold conditions  
- **EMA50 & EMA200**: Trend-following indicators (Golden Cross & Death Cross)  
- **ATR**: Measures absolute volatility  
- **nATR**: Compares volatility relative to price  

## 📌 Installation  
```bash
git clone https://github.com/yourusername/your-repo.git
