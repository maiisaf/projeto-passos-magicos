"""
Classes de transformação do pipeline preditivo de risco de defasagem.

"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LimpadorBase(BaseEstimator, TransformerMixin):
    """
    Extrai Fase_num a partir do texto da coluna Fase.
    'Alfa' → 0 | 'Fase N' → N
    fit() registra fases conhecidas no treino.
    transform() avisa se chegar fase desconhecida.
    """

    def fit(self, X, y=None):
        self.fases_conhecidas_ = set(X['Fase'].dropna().unique())
        return self

    def transform(self, X):
        X = X.copy()

        def _fase_para_numero(valor):
            if pd.isna(valor):
                return np.nan
            v = str(valor).strip().upper()
            if 'ALFA' in v:
                return 0
            match = re.search(r'\d', v)
            if match:
                return int(match.group())
            return np.nan

        X['Fase_num'] = X['Fase'].apply(_fase_para_numero)

        desconhecidos = set(X['Fase'].dropna().unique()) - self.fases_conhecidas_
        if desconhecidos:
            print(f"  ⚠️  Fases não vistas no treino: {desconhecidos} → virarão NaN")

        return X


class ImputadorIndicadores(BaseEstimator, TransformerMixin):
    """
    Imputa nulos em IDA, IEG, IAA, IPS, IPV com a mediana do grupo
    Fase + Ano_avaliacao aprendida no treino.
    Fallback: mediana global para grupos não vistos.
    """

    def __init__(self, indicadores=None):
        self.indicadores = indicadores or ['IDA', 'IEG', 'IAA', 'IPS', 'IPV']

    def fit(self, X, y=None):
        self.medianas_grupo_ = {}
        self.medianas_global_ = {}

        for col in self.indicadores:
            if col not in X.columns:
                continue
            self.medianas_grupo_[col] = (
                X.groupby(['Fase', 'Ano_avaliacao'])[col]
                .median()
                .to_dict()
            )
            self.medianas_global_[col] = X[col].median()

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.indicadores:
            if col not in X.columns:
                continue
            if X[col].isna().sum() == 0:
                continue

            def _imputa(row, col=col):
                if pd.notna(row[col]):
                    return row[col]
                chave = (row['Fase'], row['Ano_avaliacao'])
                return self.medianas_grupo_[col].get(
                    chave, self.medianas_global_[col]
                )

            X[col] = X.apply(_imputa, axis=1)
            X[col] = X[col].fillna(self.medianas_global_[col])

        return X


class ImputadorIPP(BaseEstimator, TransformerMixin):
    """
    1. Cria flag ipp_disponivel (1 = tinha dado original)
    2. Imputa IPP com mediana por Fase aprendida no treino
    3. Fallback global para Fases sem dado (ex: 8 e 9)
    """

    def fit(self, X, y=None):
        self.mediana_por_fase_ = (
            X.groupby('Fase')['IPP']
            .median()
            .to_dict()
        )
        self.mediana_global_ = X['IPP'].median()
        return self

    def transform(self, X):
        X = X.copy()
        X['ipp_disponivel'] = X['IPP'].notna().astype(int)

        def _imputa_ipp(row):
            if pd.notna(row['IPP']):
                return row['IPP']
            mediana_fase = self.mediana_por_fase_.get(row['Fase'], np.nan)
            if pd.notna(mediana_fase):
                return mediana_fase
            return self.mediana_global_

        X['IPP'] = X.apply(_imputa_ipp, axis=1)
        return X


class EncoderCategorico(BaseEstimator, TransformerMixin):
    """
    Encoding ordinal de Pedra_atual e nominal de Persona_aluno.
    fit() registra categorias conhecidas + aprende mediana da Pedra.
    transform() avisa sobre categorias novas.
    Categorias desconhecidas → NaN.
    """

    MAPA_PEDRA = {
        'Quartzo'     : 1,
        'Ágata'       : 2,
        'Ametista'    : 3,
        'Topázio'     : 4,
        'Sem registro': np.nan,
    }

    MAPA_PERSONA = {
        'Alto desempenho (estrelas resilientes)' : 0,
        'Alerta acadêmico (desfocados)'          : 1,
        'Esforçados em risco emocional'          : 2,
        'Crise de autopercepção (invisíveis)'    : 3,
        'Dados incompletos para perfil'          : np.nan,
    }

    def fit(self, X, y=None):
        self.pedras_conhecidas_   = set(X['Pedra_atual'].dropna().unique())
        self.personas_conhecidas_ = set(X['Persona_aluno'].dropna().unique())
        pedra_temp = X['Pedra_atual'].map(self.MAPA_PEDRA)
        self.mediana_pedra_ = pedra_temp.median()
        return self

    def transform(self, X):
        X = X.copy()

        desconhecidas_pedra = (
            set(X['Pedra_atual'].dropna().unique()) - self.pedras_conhecidas_
        )
        if desconhecidas_pedra:
            print(f"  ⚠️  Pedras não vistas no treino: {desconhecidas_pedra} → NaN")

        X['Pedra_enc'] = (
            X['Pedra_atual']
            .map(self.MAPA_PEDRA)
            .fillna(self.mediana_pedra_)
        )

        desconhecidas_persona = (
            set(X['Persona_aluno'].dropna().unique()) - self.personas_conhecidas_
        )
        if desconhecidas_persona:
            print(f"  ⚠️  Personas não vistas no treino: {desconhecidas_persona} → NaN")

        X['Persona_enc'] = X['Persona_aluno'].map(self.MAPA_PERSONA)
        return X


class CriadorFeatures(BaseEstimator, TransformerMixin):
    """
    Cria 3 features derivadas que capturam relações entre indicadores.
    gap_percepcao_realidade : IAA - IDA
    indice_bem_estar        : (IPS + IAA) / 2
    pressao_psicossocial    : IEG - IPS
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['gap_percepcao_realidade'] = X['IAA'] - X['IDA']
        X['indice_bem_estar']        = (X['IPS'] + X['IAA']) / 2
        X['pressao_psicossocial']    = X['IEG'] - X['IPS']
        return X


class SeletorColunas(BaseEstimator, TransformerMixin):
    """
    Garante que o DataFrame final tem exatamente as features
    que o modelo espera, na ordem correta.
    """

    FEATURES = [
        'IDA', 'IEG', 'IAA', 'IPS', 'IPV', 'IPP',
        'Nota_mat', 'Nota_port',
        'Idade', 'Fase_num', 'Pedra_enc',
        'ipp_disponivel',
        'Delta_IDA', 'Delta_IEG', 'Delta_IPS', 'Delta_IAA', 'Delta_IPV',
        'tem_historico_IDA',
        'gap_percepcao_realidade', 'indice_bem_estar', 'pressao_psicossocial',
        'Persona_enc',
    ]

    def fit(self, X, y=None):
        self.features_disponiveis_ = [f for f in self.FEATURES if f in X.columns]
        ausentes = [f for f in self.FEATURES if f not in X.columns]
        if ausentes:
            print(f"  ⚠️  Features ausentes: {ausentes}")
        return self

    def transform(self, X):
        return X[self.features_disponiveis_].copy()

    def get_feature_names_out(self):
        return self.features_disponiveis_