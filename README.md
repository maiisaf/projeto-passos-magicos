Datathon Passos Mágicos — POSTECH
Projeto de analytics educacional desenvolvido para o Datathon da Associação Passos Mágicos, com o objetivo de identificar alunos em risco de defasagem escolar e apoiar a tomada de decisão da equipe pedagógica.

Aplicativo em produção:
🔗 projeto-p-magicos-iagnvpwys3az9mwxsdsn2k.streamlit.app

O app permite:
Visualizar o dashboard analítico do programa (Tableau Public)
Calcular a probabilidade de risco de defasagem de um aluno individualmente
Analisar múltiplos alunos em lote via upload de CSV

📁 Estrutura do repositório:

projeto-passos-magicos/
│
├── aplicativo/                        # App Streamlit
│   ├── app.py                         # Aplicativo principal
│   ├── transformers.py                # Classes do pipeline de ML
│   ├── requirements.txt               # Dependências do projeto
│   ├── logo.png                       # Logo da Associação Passos Mágicos
│   └── .streamlit/
│       └── config.toml                # Tema e configurações visuais
│
├── dados_notebooks/                   # Análises e modelagem
│   ├── exploracao_dados.ipynb         # Análise exploratória dos dados
│   ├── modelo_ml.ipynb                # Notebook de modelagem preditiva
│   ├── base_dados_passos_magicos.csv  # Base de dados PEDE 2022–2024
│   ├── graficos/                      # Gráficos gerados pelo modelo
│   └── modelo/
│       ├── pipeline_risco.joblib      # Pipeline treinado (transformers + modelo)
│       └── artefatos_modelo.json      # Métricas, features e metadados do modelo
│
├── Documentacao_Tecnica.pdf           # Documentação técnica completa do projeto
├── Apresentacao.pdf                   # Apresentação em PowerPoint dos principais insights
├── .gitignore
└── README.md
