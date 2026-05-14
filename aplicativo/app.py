"""
app.py — Passos Mágicos
Aplicativo preditivo de risco de defasagem escolar.
"""

import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from transformers import (
    LimpadorBase, ImputadorIndicadores, ImputadorIPP,
    EncoderCategorico, CriadorFeatures, SeletorColunas
)

st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cores ──────────────────────────────────────────────────
azul_escuro  = "#1B3A6B"
azul_medio   = "#2E5FA3"
azul_claro   = "#E8EEF7"
gauge_baixo  = "#B2C8E9"
gauge_medio  = "#7C90B2"
gauge_alto   = "#004580"
graf_verm    = "#D41317"
graf_azul    = "#004580"
cinza_texto  = "#4A4A4A"
cinza_leve   = "#F5F7FA"
branco       = "#FFFFFF"

# ── Nomes formatados dos indicadores ───────────────────────
NOMES = {
    "IDA"                     : "Desempenho Acadêmico",
    "IEG"                     : "Engajamento",
    "IAA"                     : "Autoavaliação",
    "IPS"                     : "Psicossocial",
    "IPV"                     : "Ponto de Virada",
    "IPP"                     : "Psicopedagógico",
    "Fase_num"                : "Fase",
    "Pedra_enc"               : "Pedra",
    "Idade"                   : "Idade",
    "Nota_mat"                : "Nota Matemática",
    "Nota_port"               : "Nota Português",
    "gap_percepcao_realidade" : "Gap Percepção/Realidade",
    "indice_bem_estar"        : "Índice Bem-Estar",
    "pressao_psicossocial"    : "Pressão Psicossocial",
    "ipp_disponivel"          : "IPP Disponível",
    "tem_historico_IDA"       : "Histórico IDA",
    "Persona_enc"             : "Perfil",
    "Ano_avaliacao"           : "Ano",
    "Delta_IDA"               : "Δ IDA",
    "Delta_IEG"               : "Δ IEG",
    "Delta_IPS"               : "Δ IPS",
    "Delta_IAA"               : "Δ IAA",
    "Delta_IPV"               : "Δ IPV",
}

def fmt(feat):
    return NOMES.get(feat, feat)

# ── CSS ────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {branco}; }}

    h2 {{ color: {azul_escuro} !important; font-weight: 700 !important; }}
    h3 {{ color: {azul_medio}  !important; font-weight: 600 !important; }}
    p, li {{ color: {cinza_texto}; }}

    /* Seção numerada */
    .secao-header {{
        font-size: 20px; font-weight: 550;
        color: {branco};
        background: {azul_escuro};
        padding: 6px 14px; border-radius: 6px;
        display: block; margin-bottom: 12px; margin-top: 4px;
    }}

    /* Divisor entre seções */
    .divisor-secao {{
        border: none;
        border-top: 1px solid {azul_claro};
        margin: 16px 0 14px 0;
    }}

    /* KPI cards */
    .kpi-card {{
        background: {azul_claro};
        border-left: 4px solid {azul_medio};
        border-radius: 8px; padding: 14px 18px;
    }}
    .kpi-label {{
        font-size: 11px; color: {cinza_texto};
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .kpi-valor {{
        font-size: 26px; font-weight: 700;
        color: {azul_escuro}; line-height: 1.2;
    }}

    /* Alertas de risco (tons de azul) */
    .risco-alto {{
        background: #E8F0FA;
        border-left: 5px solid {gauge_alto};
        border-radius: 8px; padding: 16px 20px;
    }}
    .risco-medio {{
        background: #EDF1F7;
        border-left: 5px solid {gauge_medio};
        border-radius: 8px; padding: 16px 20px;
    }}
    .risco-baixo {{
        background: #F3F7FC;
        border-left: 5px solid {gauge_baixo};
        border-radius: 8px; padding: 16px 20px;
    }}

    .divisor {{
        border: none; border-top: 4px solid {azul_claro}; margin: 20px 0;
    }}

    .stButton > button {{
        background-color: {azul_medio} !important;
        color: white !important; border: none !important;
        border-radius: 6px !important; font-weight: 600 !important;
        font-size: 15px !important; padding: 10px 28px !important; width: 100%;
    }}
    .stButton > button:hover {{ background-color: {azul_escuro} !important; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ background-color: {gauge_baixo}; }}
    [data-testid="stSidebar"] * {{ color: {cinza_texto} !important; }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {cinza_texto} !important; font-size: 15px; padding: 6px 0;
    }}
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
        color: {cinza_texto} !important;
    }}
    /* Radio button color via accent-color */
    [data-testid="stSidebar"] input[type="radio"] {{
        accent-color: {azul_medio};
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Carregamento ───────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELO_DIR = os.path.join(BASE_DIR, "..", "dados_notebooks", "modelo")
 
@st.cache_resource
def carregar_pipeline():
    return joblib.load(os.path.join(MODELO_DIR, "pipeline_risco_1.joblib"))
 
@st.cache_data
def carregar_artefatos():
    with open(os.path.join(MODELO_DIR, "artefatos_modelo_1.json"), "r", encoding="utf-8") as f:
        return json.load(f)
 
try:
    pipeline  = carregar_pipeline()
    artefatos = carregar_artefatos()
except FileNotFoundError as e:
    st.error(f"Arquivo nao encontrado: {e}")
    st.stop()


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(BASE_DIR, "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown(
        f"<hr style='border-color: rgba(255,255,255,0.25); margin: 14px 0'>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='color:{branco}; font-size:15px; font-weight:700; margin:0 0 16px 0'>Predicao de Risco de Defasagem</p>",
        unsafe_allow_html=True
    )
    pagina = st.radio(
        "Navegacao",
        ["📊  Dashboard", "🔍  Analise Individual", "📁  Analise em Lote"],
        label_visibility="collapsed",
    )


# ── Funções ────────────────────────────────────────────────
def calcular_risco(dados_aluno: dict):
    df_input = pd.DataFrame([dados_aluno])
    prob     = float(pipeline.predict_proba(df_input)[0][1])

    modelo_final = pipeline.named_steps["modelo"]
    features     = pipeline.named_steps["seletor"].features_disponiveis_
    X_transf     = pipeline[:-1].transform(df_input)
    X_df         = pd.DataFrame(X_transf, columns=features)

    explainer   = shap.TreeExplainer(modelo_final)
    shap_values = explainer.shap_values(X_df)
    if hasattr(shap_values, "__len__") and len(shap_values) == 2:
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    contribuicoes = dict(zip(features, sv))
    return prob, contribuicoes


def gauge_risco(prob: float):
    fig, ax = plt.subplots(figsize=(4.5, 2.8), facecolor="none")
    ax.set_facecolor("none")

    zonas = [
        (np.linspace(np.pi, np.pi * 0.60, 100), gauge_baixo),
        (np.linspace(np.pi * 0.60, np.pi * 0.35, 100), gauge_medio),
        (np.linspace(np.pi * 0.35, 0, 100), gauge_alto),
    ]
    for thetas, cor in zonas:
        x_e = np.cos(thetas); y_e = np.sin(thetas)
        x_i = 0.65 * np.cos(thetas[::-1]); y_i = 0.65 * np.sin(thetas[::-1])
        ax.fill(np.concatenate([x_e, x_i]), np.concatenate([y_e, y_i]),
                color=cor, alpha=0.92)

    angulo = np.pi * (1 - prob)
    ax.annotate("", xy=(0.80 * np.cos(angulo), 0.80 * np.sin(angulo)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=azul_escuro,
                                lw=2.5, mutation_scale=16))
    ax.add_patch(plt.Circle((0, 0), 0.07, color=azul_escuro, zorder=5))
    ax.text(0, -0.18, f"{prob:.0%}", ha="center", fontsize=24,
            fontweight="bold", color=azul_escuro)
    ax.text(0, -0.36, "probabilidade de risco", ha="center",
            fontsize=9, color=cinza_texto)
    ax.text(-1.05, -0.10, "Baixo", ha="center", fontsize=8, color=cinza_texto)
    ax.text(0,      1.08, "Medio", ha="center", fontsize=8, color=cinza_texto)
    ax.text( 1.05, -0.10, "Alto",  ha="center", fontsize=8, color=cinza_texto)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.55, 1.25); ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=False)
    plt.close()


def grafico_contribuicoes(contribuicoes: dict):
    serie = (pd.Series({fmt(k): v for k, v in contribuicoes.items()})
               .sort_values(key=abs, ascending=True).tail(10))
    cores = [graf_verm if v > 0 else graf_azul for v in serie.values]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="none")
    ax.set_facecolor(cinza_leve)
    ax.barh(range(len(serie)), serie.values, color=cores, height=0.6)
    ax.set_yticks(range(len(serie)))
    ax.set_yticklabels(serie.index, fontsize=10, color=cinza_texto)
    ax.axvline(0, color=azul_escuro, lw=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Contribuição para o risco", color=cinza_texto, fontsize=10)
    ax.tick_params(colors=cinza_texto)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    legend = [mpatches.Patch(color=graf_verm, label="Aumenta risco"),
              mpatches.Patch(color=graf_azul,  label="Reduz risco")]
    ax.legend(handles=legend, fontsize=9, framealpha=0.9, edgecolor="none")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════
# PÁGINA 1 — ANÁLISE INDIVIDUAL
# ══════════════════════════════════════════════════════════
if "Dashboard" in pagina:
 
    st.markdown("##### Visão analítica completa do programa Passos Mágicos.")
    st.markdown("<hr class='divisor'>", unsafe_allow_html=True)
 
    # Workbook completa — navegação entre abas feita pelo próprio Tableau
    tableau_html = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>body{margin:0;padding:0;}</style></head>
<body>
<div class='tableauPlaceholder' id='viz1778768134275' style='position:relative'>
    <noscript>
        <a href='#'>
            <img alt='Visao Geral'
                 src='https://public.tableau.com/static/images/da/dashboard_passos_magicos/VisoGeral/1_rss.png'
                 style='border:none' />
        </a>
    </noscript>
    <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='dashboard_passos_magicos/VisoGeral' />
        <param name='tabs' value='yes' />
        <param name='toolbar' value='yes' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='pt-BR' />
    </object>
</div>
<script type='text/javascript'>
    var divElement = document.getElementById('viz1778768134275');
    var vizElement = divElement.getElementsByTagName('object')[0];
    vizElement.style.width  = '100%';
    vizElement.style.height = '857px';
    var scriptElement = document.createElement('script');
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
</body>
</html>"""
 
    st.components.v1.html(tableau_html, height=860, scrolling=False)

elif "Individual" in pagina:

    st.markdown("##### Preencha os indicadores do aluno para calcular a probabilidade de risco de defasagem.")
    st.markdown("<hr class='divisor'>", unsafe_allow_html=True)

    with st.form("form_aluno"):

        # ── Seção 1: Contexto ──────────────────────────────
        st.markdown("<span class='secao-header'>1. Contexto</span>",
                    unsafe_allow_html=True)

        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            fase = st.selectbox("Fase", [
                "Selecione uma opção",
                "Alfa","Fase 1","Fase 2","Fase 3","Fase 4",
                "Fase 5","Fase 6","Fase 7","Fase 8","Fase 9"
            ])
        with cc2:
            pedra = st.selectbox("Pedra atual", [
                "Selecione uma opção",
                "Quartzo","Ágata","Ametista","Topázio",
            ])
        with cc3:
            idade = st.number_input("Idade (12)", 6, 25, value=None, step=1, placeholder="Digite um número")
        with cc4:
            st.markdown("")   # espaço visual

        st.markdown("<hr class='divisor-secao'>", unsafe_allow_html=True)

        # ── Seção 2: Indicadores ───────────────────────────
        st.markdown("<span class='secao-header'>2. Indicadores</span>",
                    unsafe_allow_html=True)

        i1, i2, i3, i4 = st.columns(4)
        with i1:
            ida = st.number_input("IDA — Desempenho Acadêmico (0.0 a 10.0)", 0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")
            iaa = st.number_input("IAA — Autoavaliação (0.0 a 10.0)", 0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")
        with i2:
            ieg = st.number_input("IEG — Engajamento (0.0 a 10.0)",0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")
            ips = st.number_input("IPS — Psicossocial (0.0 a 10.0)",0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")
        with i3:
            ipv = st.number_input("IPV — Ponto de Virada (0.0 a 10.0)", 0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")
            ipp = st.number_input("IPP — Psicopedagógico (0.0 a 10.0)",0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")
        with i4:
            nota_mat  = st.number_input("Nota Matemática (0.0 a 10.0)",0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")
            nota_port = st.number_input("Nota Português (0.0 a 10.0)",0.0, 10.0, value=None, step=0.1, placeholder = "Digite um número")

        st.markdown("<hr class='divisor-secao'>", unsafe_allow_html=True)

        # ── Perfil ─────────────────────────────────────────
        persona = "Dados incompletos para perfil"


        # Deltas fixos como NaN
        delta_ida = delta_ieg = delta_ips = delta_iaa = delta_ipv = np.nan
        tem_historico = 0

        st.markdown("")
        calcular = st.form_submit_button(
            "🔍  Calcular probabilidade de risco",
            use_container_width=True,
        )

    if calcular:
        campos_vazios = [nome for nome, val in { "Idade":idade,
            "IDA": ida, "IEG": ieg, "IAA": iaa,
            "IPS": ips, "IPV": ipv, "IPP": ipp,
            "Nota Matemática": nota_mat, "Nota Português": nota_port,
        }.items() if val is None]

        if fase == "Selecione uma opção" or pedra == "Selecione uma opção":
            st.warning("Por favor, selecione a Fase e a Pedra atual do aluno.")
        
        elif campos_vazios:
            st.warning(f"Por favor, preencha os campos: {', '.join(campos_vazios)}")

        else:
            dados = {
                "Fase": fase, "Pedra_atual": pedra,
                "Persona_aluno": persona,
                "Idade": idade, "Ano_avaliacao": 2024,
                "Nota_mat": nota_mat, "Nota_port": nota_port,
                "IDA": ida, "IEG": ieg, "IAA": iaa,
                "IPS": ips, "IPV": ipv, "IPP": ipp,
                "Delta_IDA": delta_ida, "Delta_IEG": delta_ieg,
                "Delta_IPS": delta_ips, "Delta_IAA": delta_iaa,
                "Delta_IPV": delta_ipv,
                "tem_historico_IDA": int(tem_historico),
            }

            with st.spinner("Calculando..."):
                prob, contribuicoes = calcular_risco(dados)

            st.markdown("<hr class='divisor'>", unsafe_allow_html=True)
            col_gauge, col_resultado = st.columns([1, 2])

            with col_gauge:
                gauge_risco(prob)

            with col_resultado:
                if prob >= 0.65:
                    classe = "risco-alto"; icone = "🔵"; nivel = "Risco Alto de Defasagem"
                    cor_nivel = gauge_alto
                    msg = (f"O modelo estima <b>{prob:.0%} de probabilidade</b> de este aluno "
                           f"entrar em defasagem. Recomenda-se atenção <b>prioritaria</b> da "
                           f"equipe pedagógica e psicossocial.")
                elif prob >= 0.40:
                    classe = "risco-medio"; icone = "🔵"; nivel = "Risco Moderado"
                    cor_nivel = gauge_medio
                    msg = (f"Probabilidade de <b>{prob:.0%}</b>. O aluno apresenta alguns "
                           f"sinais de alerta. Monitoramento preventivo recomendado.")
                else:
                    classe = "risco-baixo"; icone = "🔵"; nivel = "Risco Baixo"
                    cor_nivel = gauge_baixo
                    msg = (f"Probabilidade de <b>{prob:.0%}</b>. Os indicadores estão "
                           f"dentro do esperado para a fase atual.")

                st.markdown(f"""
                <div class='{classe}'>
                    <b style='font-size:17px; color:{cor_nivel}'>{nivel}</b>
                    <br><br>{msg}
                </div>""", unsafe_allow_html=True)

                st.markdown("")
                st.markdown("**Principais fatores para este aluno:**")

                contrib_ord = pd.Series(contribuicoes).sort_values(key=abs, ascending=False)
                for feat, val in contrib_ord.head(5).items():
                    cor_f = graf_verm if val > 0 else graf_azul
                    sinal = "aumenta o risco" if val > 0 else "reduz o risco"
                    arrow = "↑" if val > 0 else "↓"
                    # Mostra só a sigla nos fatores
                    sigla = feat.upper() if feat in ["ida","ieg","iaa","ips","ipv","ipp"] else feat
                    st.markdown(
                        f"<span style='color:{cor_f}; font-weight:600'>{arrow} {sinal}</span>"
                        f" — <b>{fmt(feat)}</b>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<hr class='divisor'>", unsafe_allow_html=True)
            st.markdown("##### Contribuição detalhada por indicador")
            st.caption("Vermelho = aumenta o risco  |  Azul = fator de protecao")
            grafico_contribuicoes(contribuicoes)


# ══════════════════════════════════════════════════════════
# PÁGINA 2 — ANÁLISE EM LOTE
# ══════════════════════════════════════════════════════════
elif "Lote" in pagina:

    st.markdown("##### Faça upload de um CSV com vários alunos para calcular o risco de cada um.")
    st.markdown("<hr class='divisor'>", unsafe_allow_html=True)

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown("""
        **Colunas obrigatórias:**
        `Fase` · `Pedra_atual` · `Persona_aluno` · `Idade`
        · `IDA` · `IEG` · `IAA` · `IPS` · `IPV` · `IPP` · `Nota_mat` · `Nota_port`

        Separador: **ponto-e-vírgula** (`;`)
        """)
    with col_btn:
        template = pd.DataFrame([{
            "Fase": "Fase 3", "Pedra_atual": "Ágata",
            "Persona_aluno": "Alerta acadêmico (desfocados)",
            "Idade": 12,
            "IDA": 5.5, "IEG": 7.0, "IAA": 8.0,
            "IPS": 5.0, "IPV": 6.5, "IPP": 7.5,
            "Nota_mat": 5.0, "Nota_port": 6.0,
        }])
        st.download_button("Baixar template",
            template.to_csv(index=False, sep=";").encode("utf-8"),
            "template_alunos.csv", "text/csv", use_container_width=True)

    st.markdown("")
    arquivo = st.file_uploader("Selecione o arquivo CSV", type=["csv"],
                               label_visibility="collapsed")

    if arquivo:
        try:
            df_up = pd.read_csv(arquivo, sep=";")
            st.success(f"✅  {len(df_up)} alunos carregados.")

            for col in ["Delta_IDA","Delta_IEG","Delta_IPS","Delta_IAA","Delta_IPV"]:
                if col not in df_up.columns:
                    df_up[col] = np.nan
            if "tem_historico_IDA" not in df_up.columns:
                df_up["tem_historico_IDA"] = df_up["Delta_IDA"].notna().astype(int)
            if "Persona_aluno" not in df_up.columns:
                df_up["Persona_aluno"] = "Dados incompletos para perfil"

            with st.spinner("Calculando probabilidades..."):
                probs = pipeline.predict_proba(df_up)[:, 1]

            df_res = df_up.copy()
            df_res["Probabilidade de Risco"] = (probs * 100).round(1).astype(str) + "%"
            df_res["Risco"] = pd.cut(
                probs, bins=[0, 0.40, 0.65, 1.0],
                labels=["🟢 Baixo", "🟡 Moderado", "🔴 Alto"],
            )
            df_res["_prob_sort"] = probs
            df_res = df_res.sort_values("_prob_sort", ascending=False).drop(columns=["_prob_sort"])

            st.markdown("<hr class='divisor'>", unsafe_allow_html=True)

            n_alto = int((probs >= 0.65).sum())
            n_mod  = int(((probs >= 0.40) & (probs < 0.65)).sum())
            n_bx   = int((probs < 0.40).sum())
            total  = len(probs)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""
                <div class='kpi-card' style='border-left-color:{gauge_alto}'>
                    <div class='kpi-label'>Risco Alto</div>
                    <div class='kpi-valor' style='color:{gauge_alto}'>{n_alto}</div>
                    <div style='font-size:12px;color:{cinza_texto}'>{n_alto/total:.0%} dos alunos</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class='kpi-card' style='border-left-color:{gauge_medio}'>
                    <div class='kpi-label'>Risco Moderado</div>
                    <div class='kpi-valor' style='color:{gauge_medio}'>{n_mod}</div>
                    <div style='font-size:12px;color:{cinza_texto}'>{n_mod/total:.0%} dos alunos</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class='kpi-card' style='border-left-color:{gauge_baixo}'>
                    <div class='kpi-label'>Risco Baixo</div>
                    <div class='kpi-valor' style='color:{gauge_baixo}'>{n_bx}</div>
                    <div style='font-size:12px;color:{cinza_texto}'>{n_bx/total:.0%} dos alunos</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            cols_exibir = [c for c in
                ["Fase","Pedra_atual","Idade","IDA","IEG","IPS","IPV",
                 "Probabilidade de Risco","Risco"]
                if c in df_res.columns]

            st.dataframe(df_res[cols_exibir].reset_index(drop=True),
                         use_container_width=True, height=400)

            st.download_button("Baixar resultados",
                df_res.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig"),
                "resultado_risco_alunos.csv", "text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            st.info("Verifique se o separador e ponto-e-vírgula (;) e se as colunas estão no formato correto.")