# 🚢 Titanic — Sobrevivência com Machine Learning

> **Classificação binária com feature engineering avançado, seleção de modelos e interpretabilidade de resultados.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

---

## 🎯 Objetivo

Projeto de **classificação binária** para prever a sobrevivência de passageiros do Titanic. Vai além do tutorial básico: foca em **feature engineering criativo**, **tratamento rigoroso de dados faltantes**, **validação cruzada** e **interpretabilidade do modelo** com SHAP values — habilidades aplicáveis diretamente em problemas reais de churn, risco de crédito e diagnóstico médico.

---

## 🛠️ Stack Tecnológica

| Camada | Tecnologias |
|--------|------------|
| **Linguagem** | Python 3.10 |
| **ML** | Scikit-learn, XGBoost |
| **Interpretabilidade** | SHAP |
| **Dados** | Pandas, NumPy |
| **Visualização** | Matplotlib, Seaborn |
| **Ambiente** | Jupyter Notebook |

---

## 📂 Estrutura do Projeto

```
projeto-titanic/
│
├── 📁 notebooks/
│   ├── 01_eda_titanic.ipynb           # Análise exploratória completa
│   ├── 02_feature_engineering.ipynb   # Engenharia de features
│   ├── 03_model_selection.ipynb       # Comparação de algoritmos
│   └── 04_interpretability.ipynb      # SHAP values e explicabilidade
│
├── 📁 data/
│   ├── train.csv
│   └── test.csv
│
└── README.md
```

---

## 🔬 Feature Engineering

```python
# Extração de título social do nome (proxy de status social)
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don',
                                   'Dr','Major','Rev','Sir','Jonkheer'], 'Rare')

# Tamanho da família
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Deck extraído do número do camarote
df['Deck'] = df['Cabin'].str[0].fillna('U')
```

---

## 📊 Resultados dos Modelos

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|:---:|:---:|:---:|
| Logistic Regression | 0.793 | 0.761 | 0.851 |
| Random Forest | 0.834 | 0.809 | 0.903 |
| Gradient Boosting | **0.851** | **0.827** | **0.921** |
| SVM (RBF) | 0.829 | 0.800 | 0.898 |

> **Cross-Validation (5-fold) — Gradient Boosting: 84.2% ± 1.8%**

---

## 📌 Principais Insights

- 👩 **Gênero** é o preditor mais poderoso — mulheres tiveram taxa de sobrevivência de 74% vs 19% dos homens
- 🎫 **Classe da cabine** é o segundo fator mais relevante (1ª classe: 63% | 3ª classe: 24%)
- 👶 **Crianças (< 12 anos)** tiveram taxa de sobrevivência 55% maior que adultos
- 🃏 Feature "Title" criada via NLP simples aumentou o F1 em +4.2%

---

## 🚀 Como Executar

```bash
git clone https://github.com/Jk-Pascoal/projeto-titanic.git
cd projeto-titanic

pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter

jupyter notebook notebooks/01_eda_titanic.ipynb
```

---

## 📬 Contato

**Jakson Pascoal** | [LinkedIn](https://linkedin.com/in/jakson-pascoal) | [GitHub](https://github.com/Jk-Pascoal)
