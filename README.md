# Datathon Passos Mágicos — POSTECH

Projeto de analytics educacional para identificar alunos em risco de defasagem escolar e apoiar a equipe pedagógica da Associação Passos Mágicos.

---

## Aplicativo em produção

🔗 **[Acessar o app](https://projeto-p-magicos-iagnvpwys3az9mwxsdsn2k.streamlit.app/)**

O app permite:
- Visualizar o dashboard analítico do programa (Tableau Public)
- Calcular a probabilidade de risco de defasagem individualmente
- Analisar múltiplos alunos em lote via upload de CSV

---

## 📁 Estrutura do repositório

**aplicativo/:**
- `app.py` — aplicativo principal Streamlit
- `transformers.py` — classes do pipeline de ML
- `requirements.txt` — dependências do projeto
- `logo.png` — logo da Associação Passos Mágicos
- `.streamlit/config.toml` — tema e configurações visuais

**dados_notebooks/:**
- `exploracao_dados.ipynb` — análise exploratória dos dados
- `modelo_ml.ipynb` — notebook de modelagem preditiva
- `base_dados_passos_magicos.csv` — base de dados PEDE 2022–2024
- `graficos/` — gráficos gerados pelo modelo
- `modelo/pipeline_risco.joblib` — pipeline treinado (transformers + modelo)
- `modelo/artefatos_modelo.json` — métricas, features e metadados do modelo

**Raiz:**
- `Documentacao_Tecnica.pdf` — documentação técnica completa do projeto
- `Apresentacao.pptx` — apresentação com os principais insights
- `.gitignore` — arquivos ignorados pelo git
- `README.md` — este arquivo
