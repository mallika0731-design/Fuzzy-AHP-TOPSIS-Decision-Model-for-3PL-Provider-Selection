# Hybrid Fuzzy AHP–TOPSIS Decision Model for 3PL Provider Selection

## Overview

Selecting the right Third-Party Logistics (3PL) provider is a complex decision involving multiple conflicting criteria such as cost, delivery performance, safety, technology capability, and sustainability.

This project implements a **hybrid decision science framework** that integrates:

• **Fuzzy Analytic Hierarchy Process (Fuzzy AHP)** for determining criteria weights  
• **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** for ranking alternatives  
• **Fuzzy Inference System (FIS)** for rule-based evaluation  
• **Monte Carlo Simulation** to test the robustness of the ranking results  

The objective is to create a **data-driven decision support system** for evaluating logistics providers under uncertainty.

---

# Problem Statement

Organizations must evaluate multiple logistics providers based on several operational and strategic criteria. Traditional decision approaches struggle with uncertainty and subjective judgments.

This project models the problem using **Multi-Criteria Decision Making (MCDM)** techniques combined with **fuzzy logic** to produce a systematic and robust ranking of logistics providers.

---

# Decision Criteria

The logistics providers are evaluated based on five criteria:

• Price  
• Delivery Performance  
• Safety  
• Technology Capability  
• Social Sustainability  

These criteria represent critical operational and sustainability factors influencing logistics partner selection.

---

# Methodology

The decision pipeline implemented in this project is:

Fuzzy Pairwise Comparison Matrix  
↓  
Fuzzy AHP Weight Calculation  
↓  
TOPSIS Multi-Criteria Ranking  
↓  
Fuzzy Inference System Evaluation  
↓  
Monte Carlo Robustness Testing  

---

# 1. Fuzzy AHP

Fuzzy AHP is used to determine the relative importance of each decision criterion.

A **fuzzy pairwise comparison matrix** is constructed using **triangular fuzzy numbers** to represent uncertainty in expert judgments.

Steps involved:

1. Construct fuzzy pairwise comparison matrix
2. Compute geometric mean of each row
3. Normalize fuzzy weights
4. Defuzzify weights using centroid method
5. Normalize final weights

The resulting weights represent the **relative importance of each evaluation criterion**.

---

# 2. TOPSIS

TOPSIS ranks alternatives based on their **distance from an ideal best and ideal worst solution**.

Steps performed:

1. Normalize the decision matrix
2. Multiply by criteria weights
3. Determine ideal best and ideal worst solutions
4. Compute Euclidean distance from both
5. Calculate relative closeness score

Alternatives closer to the ideal solution receive **higher scores and better rankings**.

---

# 3. Fuzzy Inference System (FIS)

A **rule-based fuzzy system** evaluates providers using linguistic rules such as:

• Low price AND high delivery → Good score  
• Medium price AND medium delivery → Average score  
• Low technology OR low social score → Poor evaluation  

Triangular membership functions define fuzzy sets for each criterion.

The fuzzy system provides an **alternative ranking based on expert rule logic**.

---

# 4. Monte Carlo Simulation

Monte Carlo simulation is used to test the **robustness of the decision model**.

Small random variations are introduced to the criteria weights and the ranking process is repeated thousands of times.

This analysis measures:

• ranking stability  
• sensitivity to weight changes  
• probability of each provider being ranked first

---

# Dataset

The dataset contains **20 logistics providers** evaluated across five decision criteria.

Each provider is represented by normalized performance scores reflecting:

• cost efficiency  
• delivery reliability  
• operational safety  
• technological capability  
• social responsibility

---

# Key Outputs

The system generates:

• Fuzzy AHP criteria weights  
• TOPSIS ranking of logistics providers  
• Fuzzy Inference System scores  
• Ranking comparison between methods  
• Correlation analysis between models  
• Monte Carlo robustness probabilities  
• Data visualizations and charts

---

# Example Output

Example TOPSIS ranking:

| Provider | TOPSIS Score | Rank |
|---------|-------------|------|
| 3PL-19 | 0.885 | 1 |
| 3PL-4 | 0.845 | 2 |
| 3PL-8 | 0.836 | 3 |

Higher scores indicate closer proximity to the **ideal logistics provider**.

---

# Technologies Used

Python libraries used in this project:

• NumPy  
• Pandas  
• Matplotlib  
• Seaborn  
• scikit-fuzzy  
• openpyxl  

---

# Project Structure

```
project/
│
├── data/
├── outputs/
├── visualizations/
├── decision_science.py
├── presentation/
└── README.md
```

---

# How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the main analysis script:

```
python decision_science.py
```

Outputs and visualizations will be generated automatically.

---

# Applications

This decision framework can be applied to:

• Supplier selection  
• Vendor evaluation  
• Supply chain partner ranking  
• Sustainable procurement decisions  
• Multi-criteria business decision problems

---
## Visualizations

The system generates several visualizations including:

- Criteria weight distribution
- TOPSIS ranking bar chart
- Score comparison plots
- Correlation analysis
- Monte Carlo robustness probabilities

# License

This project is intended for educational and research purposes.
