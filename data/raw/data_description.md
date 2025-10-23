# **Fama–French 5 Factors (2×3) — Summary**

**Data Period**  
- **Monthly Returns:** July 1963 – August 2025  
- **Annual Returns:** 1964 – 2024  

**References:**  
- Fama & French (2014), *A Five-Factor Asset Pricing Model*.  

---

## **1. Construction Overview**

The **Fama–French 5 factors (2×3)** are built using 6×3 portfolios sorted by:
- **Size** (Small / Big)
- **Book-to-Market (B/M)**
- **Operating Profitability (OP)**
- **Investment (INV)**  

Each factor is constructed as an average of portfolio returns along these dimensions.

---

## **2. Factor Definitions**

### **SMB — Small Minus Big**
Average return on **small** stock portfolios minus **big** stock portfolios.

\[
\text{SMB(B/M)} = \frac{1}{3}(\text{Small Value + Small Neutral + Small Growth}) - \frac{1}{3}(\text{Big Value + Big Neutral + Big Growth})
\]

\[
\text{SMB(OP)} = \frac{1}{3}(\text{Small Robust + Small Neutral + Small Weak}) - \frac{1}{3}(\text{Big Robust + Big Neutral + Big Weak})
\]

\[
\text{SMB(INV)} = \frac{1}{3}(\text{Small Conservative + Small Neutral + Small Aggressive}) - \frac{1}{3}(\text{Big Conservative + Big Neutral + Big Aggressive})
\]

\[
\boxed{\text{SMB} = \frac{1}{3}[\text{SMB(B/M)} + \text{SMB(OP)} + \text{SMB(INV)}]}
\]

---

### **HML — High Minus Low**
Measures the **value premium** (value stocks outperforming growth stocks):

\[
\boxed{\text{HML} = \frac{1}{2}(\text{Small Value + Big Value}) - \frac{1}{2}(\text{Small Growth + Big Growth})}
\]

---

### **RMW — Robust Minus Weak**
Measures the **profitability premium**:

\[
\boxed{\text{RMW} = \frac{1}{2}(\text{Small Robust + Big Robust}) - \frac{1}{2}(\text{Small Weak + Big Weak})}
\]

---

### **CMA — Conservative Minus Aggressive**
Measures the **investment premium**:

\[
\boxed{\text{CMA} = \frac{1}{2}(\text{Small Conservative + Big Conservative}) - \frac{1}{2}(\text{Small Aggressive + Big Aggressive})}
\]

---

### **Rm – Rf**
The **market excess return**:  
Value-weighted return on all U.S. CRSP firms (NYSE, AMEX, NASDAQ)  
minus the **1-month Treasury Bill rate**.

- **T-bill source:**  
  - Up to May 2024: *Ibbotson Associates*  
  - From June 2024: *ICE BofA US 1-Month Treasury Bill Index*

---

## **3. Stock Universe**
All **NYSE, AMEX, and NASDAQ** firms with valid:
- Market equity data (Dec *t–1* and Jun *t*)
- Positive book equity (*for SMB, HML, RMW*)
- Non-missing revenues and cost data (*for SMB, RMW*)
- Total assets (*for SMB, CMA*)

---

## **4. Usage**
These factors are the canonical inputs for:
- Asset pricing tests
- Factor regressions (e.g., Fama-MacBeth)
- Multi-factor backtests and performance attribution

**Typical model form:**
\[
R_i - R_f = \alpha + \beta_m (R_m - R_f) + s \cdot SMB + h \cdot HML + r \cdot RMW + c \cdot CMA + \epsilon
\]


