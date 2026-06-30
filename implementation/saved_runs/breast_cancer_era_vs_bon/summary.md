# Breast Cancer — ERA vs. Best-of-N (single run)

- Task: breast cancer binary classification (metric: **ROC-AUC**, higher is better)
- Model: `gemini-2.5-flash`
- Budget: N = 10 LLM calls per method

## Result
- Initial baseline ROC-AUC: **0.846777**
- ERA final best: **0.994284**
- Best-of-N final best: **0.984122**
- **Winner: ERA**

## Invalid candidates
- ERA: 0/10
- Best-of-N: 0/10

> Sandbox is insecure; toy reproduction only.
