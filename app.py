# ============================================================
# APP ANOVA + TUKEY
# Aplicativo para análise estatística de ensaios experimentais
# ============================================================
# Desenvolvido para pesquisadores sem conhecimento de programação
# Permite análise de ensaios experimentais com ANOVA e teste de Tukey
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from scipy.stats import shapiro, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="ANOVA e Tukey Lab",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ANOVA e Tukey Lab")

st.markdown(
    """
    Aplicativo para análise estatística de ensaios experimentais com:

    - ANOVA de uma via;
    - Teste post hoc de Tukey;
    - Teste de normalidade de Shapiro-Wilk;
    - Teste de homogeneidade de variâncias de Levene;
    - Gráfico dos intervalos de confiança de 95%;
    - Exportação dos resultados em Excel, PNG e PDF.
    """
)


# ============================================================
# CONSTANTES DE CONFIGURAÇÃO
# ============================================================

PRECISAO_SAIDA = 4
DPI_PNG = 600
TAMANHO_FIGURA = (9, 5.5)
FIGSIZE_WIDTH = 9
FIGSIZE_HEIGHT = 5.5


# ============================================================
# FUNÇÕES AUXILIARES DE DADOS
# ============================================================

def limpar_dados(df, coluna_grupo, coluna_resposta):
    """
    Limpa e padroniza os dados selecionados pelo usuário.
    
    Operações realizadas:
    - Remove espaços em branco
    - Converte vírgula em ponto para decimais
    - Remove linhas com valores ausentes
    - Remove grupos vazios ou inválidos
    
    Parâmetros:
        df (pd.DataFrame): Dataframe bruto da planilha
        coluna_grupo (str): Nome da coluna com grupos/tratamentos
        coluna_resposta (str): Nome da coluna com valores numéricos
        
    Retorna:
        pd.DataFrame: Dataframe limpo com colunas "Grupo" e "Resultado"
    """
    dados = df[[coluna_grupo, coluna_resposta]].copy()
    dados.columns = ["Grupo", "Resultado"]

    # Padroniza grupos como strings e remove espaços
    dados["Grupo"] = dados["Grupo"].astype(str).str.strip()

    # Converte resultado: aceita vírgula ou ponto como decimal
    dados["Resultado"] = (
        dados["Resultado"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )

    # Converte para numérico, descartando valores inválidos
    dados["Resultado"] = pd.to_numeric(dados["Resultado"], errors="coerce")

    # Remove linhas com dados ausentes
    dados = dados.dropna(subset=["Grupo", "Resultado"])

    # Remove grupos vazios ou inválidos
    dados = dados[
        (dados["Grupo"] != "") &
        (dados["Grupo"].str.lower() != "nan")
    ]

    return dados


def ler_planilha(arquivo):
    """
    Lê arquivo Excel ou CSV.
    
    Parâmetros:
        arquivo: Arquivo enviado pelo usuário (CSV ou XLSX)
        
    Retorna:
        tuple: (DataFrame, nome_aba ou None, lista_abas ou None)
    """
    try:
        if arquivo.name.endswith(".csv"):
            df = pd.read_csv(arquivo)
            return df, None, None
        else:
            xls = pd.ExcelFile(arquivo)
            return xls, None, xls.sheet_names
    except Exception as e:
        raise ValueError(f"Erro ao ler arquivo: {str(e)}")


def selecionar_aba_excel(arquivo):
    """
    Permite o usuário selecionar e retorna a aba desejada do Excel.
    
    Parâmetros:
        arquivo: Arquivo Excel enviado
        
    Retorna:
        pd.DataFrame: Dados da aba selecionada
    """
    xls = pd.ExcelFile(arquivo)
    aba = st.sidebar.selectbox("Escolha a aba da planilha", xls.sheet_names)
    df = pd.read_excel(arquivo, sheet_name=aba)
    return df


# ============================================================
# FUNÇÕES DE ANÁLISE ESTATÍSTICA
# ============================================================

def calcular_estatistica_descritiva(dados):
    """
    Calcula estatística descritiva por grupo.
    
    Parâmetros:
        dados (pd.DataFrame): Dataframe com colunas "Grupo" e "Resultado"
        
    Retorna:
        pd.DataFrame: Tabela com N, Média, Desvio Padrão e Erro Padrão
    """
    descritiva = dados.groupby("Grupo")["Resultado"].agg(
        N="count",
        Média="mean",
        Desvio_Padrão="std",
        Erro_Padrão=lambda x: x.std() / np.sqrt(len(x))
    ).reset_index()

    # Arredonda para a precisão definida
    for col in ["Média", "Desvio_Padrão", "Erro_Padrão"]:
        descritiva[col] = descritiva[col].round(PRECISAO_SAIDA)

    return descritiva


def executar_anova(dados):
    """
    Executa ANOVA de uma via.
    
    Parâmetros:
        dados (pd.DataFrame): Dataframe com colunas "Grupo" e "Resultado"
        
    Retorna:
        tuple: (modelo, tabela_anova_formatada, valores_f_p)
    """
    modelo = ols("Resultado ~ C(Grupo)", data=dados).fit()
    anova = sm.stats.anova_lm(modelo, typ=2)

    # Extrai valores da ANOVA
    sq_fator = anova.loc["C(Grupo)", "sum_sq"]
    gl_fator = anova.loc["C(Grupo)", "df"]
    mq_fator = sq_fator / gl_fator
    f_valor = anova.loc["C(Grupo)", "F"]
    p_valor = anova.loc["C(Grupo)", "PR(>F)"]

    sq_residuos = anova.loc["Residual", "sum_sq"]
    gl_residuos = anova.loc["Residual", "df"]
    mq_residuos = sq_residuos / gl_residuos

    # Cria tabela formatada
    anova_formatada = pd.DataFrame({
        "Fonte": ["Fator", "Resíduos"],
        "G.L.": [int(gl_fator), int(gl_residuos)],
        "SQ": [round(sq_fator, PRECISAO_SAIDA), round(sq_residuos, PRECISAO_SAIDA)],
        "MQ": [round(mq_fator, PRECISAO_SAIDA), round(mq_residuos, PRECISAO_SAIDA)],
        "F": [round(f_valor, PRECISAO_SAIDA), ""],
        "P. valor": [round(p_valor, PRECISAO_SAIDA), ""]
    })

    return modelo, anova_formatada, (f_valor, p_valor)


def avaliar_pressupostos(modelo, dados, alpha):
    """
    Avalia pressupostos de normalidade (Shapiro-Wilk) e homogeneidade (Levene).
    
    Parâmetros:
        modelo: Modelo ANOVA ajustado
        dados (pd.DataFrame): Dataframe com colunas "Grupo" e "Resultado"
        alpha (float): Nível de significância
        
    Retorna:
        pd.DataFrame: Tabela com resultados dos testes de pressupostos
    """
    # Teste de normalidade dos resíduos
    residuos = modelo.resid
    
    if len(residuos) >= 3:
        shapiro_stat, shapiro_p = shapiro(residuos)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Teste de homogeneidade de variâncias
    grupos = [
        grupo["Resultado"].values
        for nome, grupo in dados.groupby("Grupo")
    ]
    levene_stat, levene_p = levene(*grupos)

    # Interpretação dos testes
    if not np.isnan(shapiro_p):
        interpretacao_shapiro = (
            "Normalidade atendida"
            if shapiro_p >= alpha
            else "Possível violação da normalidade"
        )
    else:
        interpretacao_shapiro = "Amostra insuficiente para Shapiro-Wilk"

    interpretacao_levene = (
        "Homogeneidade atendida"
        if levene_p >= alpha
        else "Possível heterogeneidade das variâncias"
    )

    pressupostos = pd.DataFrame({
        "Teste": ["Shapiro-Wilk", "Levene"],
        "Estatística": [
            round(shapiro_stat, PRECISAO_SAIDA) if not np.isnan(shapiro_stat) else "",
            round(levene_stat, PRECISAO_SAIDA)
        ],
        "p-valor": [
            round(shapiro_p, PRECISAO_SAIDA) if not np.isnan(shapiro_p) else "",
            round(levene_p, PRECISAO_SAIDA)
        ],
        "Interpretação": [
            interpretacao_shapiro,
            interpretacao_levene
        ]
    })

    return pressupostos


def padronizar_tukey_por_ordem(tukey_df, ordem_grupos):
    """
    ⭐ FUNÇÃO CRÍTICA: Padroniza as comparações do teste de Tukey.
    
    Garante que todas as comparações seguem: MAIOR_ÍNDICE - MENOR_ÍNDICE
    onde o índice é baseado na ordem definida pelo usuário.
    
    Exemplo com ordem: ["T-REF", "T-0,4", "T-0,6", "T-0,8", "T-1,0"]
    Índices:             0        1        2        3        4
    
    As comparações esperadas são (maior índice MENOS menor índice):
    - T-1,0(4) - T-0,8(3)
    - T-1,0(4) - T-0,6(2)
    - T-0,8(3) - T-0,6(2)
    - T-1,0(4) - T-0,4(1)
    - T-0,8(3) - T-0,4(1)
    - T-0,6(2) - T-0,4(1)
    - T-1,0(4) - T-REF(0)
    - T-0,8(3) - T-REF(0)
    - T-0,6(2) - T-REF(0)
    - T-0,4(1) - T-REF(0)
    
    E NÃO comparações com índice menor primeiro como "T-REF - T-1,0".
    
    Parâmetros:
        tukey_df (pd.DataFrame): Tabela bruta de resultados do Tukey
        ordem_grupos (list): Lista de nomes dos grupos em ordem desejada
        
    Retorna:
        pd.DataFrame: Tabela padronizada com comparações no sentido correto
    """
    # Cria mapeamento de posição: grupo -> índice na ordem
    ordem = {grupo: i for i, grupo in enumerate(ordem_grupos)}

    linhas = []

    for _, row in tukey_df.iterrows():
        g1_original = str(row["Grupo 1"]).strip()
        g2_original = str(row["Grupo 2"]).strip()

        diff_original = float(row["Diferença média"])
        lower_original = float(row["IC 95% inferior"])
        upper_original = float(row["IC 95% superior"])

        pvalor = row["p-ajustado"]
        significativo = row["Significativo"]

        # Se os dois grupos estão na ordem definida, padroniza
        if g1_original in ordem and g2_original in ordem:
            idx_g1 = ordem[g1_original]
            idx_g2 = ordem[g2_original]
            
            # Determina qual grupo tem MAIOR índice
            if idx_g1 > idx_g2:
                # g1 tem maior índice: g1 - g2 é o correto
                grupo_maior = g1_original
                grupo_menor = g2_original
                diff_final = diff_original
                lower_final = lower_original
                upper_final = upper_original
                idx_maior = idx_g1
                idx_menor = idx_g2
            else:
                # g2 tem maior índice: g2 - g1 é o correto
                # Mas statsmodels retorna g1 - g2, então precisamos INVERTER
                grupo_maior = g2_original
                grupo_menor = g1_original
                diff_final = -diff_original  # Inverte o sinal
                lower_final = -upper_original  # Inverte os limites do IC
                upper_final = -lower_original
                idx_maior = idx_g2
                idx_menor = idx_g1

        else:
            # Caso um grupo não esteja na ordem definida, mantém original
            grupo_maior = g1_original
            grupo_menor = g2_original
            diff_final = diff_original
            lower_final = lower_original
            upper_final = upper_original
            idx_maior = 999  # Valor para colocar no final
            idx_menor = 999

        # Constrói a legenda: MAIOR_ÍNDICE - MENOR_ÍNDICE
        comparacao = f"{grupo_maior} - {grupo_menor}"

        linhas.append({
            "Comparação": comparacao,
            "Grupo 1": grupo_menor,
            "Grupo 2": grupo_maior,
            "Diferença média": diff_final,
            "IC 95% inferior": lower_final,
            "IC 95% superior": upper_final,
            "p-ajustado": pvalor,
            "Significativo": significativo,
            "idx_maior": idx_maior,
            "idx_menor": idx_menor
        })

    tukey_padronizado = pd.DataFrame(linhas)

    # Ordena de FORMA INVERSA para compensar o invert_yaxis() do gráfico
    # Ascending=[False, False] coloca MAIORES índices no TOPO da tabela
    # Quando gráfico inverte com invert_yaxis(), maiores índices ficam NO FINAL = correto
    tukey_padronizado = tukey_padronizado.sort_values(
        by=["idx_maior", "idx_menor"],
        ascending=[False, False]  # Maiores índices primeiro: T-1,0 no topo da tabela
    ).reset_index(drop=True)

    # Seleciona apenas as colunas necessárias
    tukey_padronizado = tukey_padronizado[
        [
            "Comparação",
            "Grupo 1",
            "Grupo 2",
            "Diferença média",
            "IC 95% inferior",
            "IC 95% superior",
            "p-ajustado",
            "Significativo"
        ]
    ]

    # Arredonda valores numéricos
    tukey_padronizado["Diferença média"] = tukey_padronizado["Diferença média"].round(PRECISAO_SAIDA)
    tukey_padronizado["IC 95% inferior"] = tukey_padronizado["IC 95% inferior"].round(PRECISAO_SAIDA)
    tukey_padronizado["IC 95% superior"] = tukey_padronizado["IC 95% superior"].round(PRECISAO_SAIDA)
    tukey_padronizado["p-ajustado"] = tukey_padronizado["p-ajustado"].round(PRECISAO_SAIDA)

    return tukey_padronizado


def executar_tukey(dados, alpha, ordem_grupos):
    """
    Executa teste post hoc de Tukey com comparações padronizadas.
    
    Parâmetros:
        dados (pd.DataFrame): Dataframe com colunas "Grupo" e "Resultado"
        alpha (float): Nível de significância
        ordem_grupos (list): Lista de nomes dos grupos em ordem desejada
        
    Retorna:
        pd.DataFrame: Tabela de Tukey padronizada
    """
    tukey = pairwise_tukeyhsd(
        endog=dados["Resultado"],
        groups=dados["Grupo"],
        alpha=alpha
    )

    # Extrai resultados para DataFrame
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    # Renomeia colunas para português
    tukey_df = tukey_df.rename(columns={
        "group1": "Grupo 1",
        "group2": "Grupo 2",
        "meandiff": "Diferença média",
        "p-adj": "p-ajustado",
        "lower": "IC 95% inferior",
        "upper": "IC 95% superior",
        "reject": "Significativo"
    })

    # ⭐ PADRONIZAÇÃO CRÍTICA: garante que as comparações estão no sentido correto
    tukey_df = padronizar_tukey_por_ordem(tukey_df, ordem_grupos)

    return tukey_df




# ============================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================

def gerar_grafico_tukey(tukey_df, nome_ensaio, unidade):
    """
    Gera gráfico dos intervalos de confiança de 95% do Tukey.
    
    Características:
    - Gráfico horizontal com barras de erro
    - Linha vertical em x=0 para referência
    - Preserva a ordem das comparações (não ordena por diferença média)
    - Comparações com MAIORES índices no topo (MATLAB style)
    
    Parâmetros:
        tukey_df (pd.DataFrame): Tabela de Tukey padronizada
        nome_ensaio (str): Nome do ensaio (para rótulo do eixo x)
        unidade (str): Unidade de medida
        
    Retorna:
        matplotlib.figure.Figure: Figura do gráfico
    """
    tukey_plot = tukey_df.copy().reset_index(drop=True)

    # Posições das comparações no eixo y (preserva ordem)
    y_pos = np.arange(len(tukey_plot))

    fig, ax = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))

    # Cria gráfico de pontos com barras de erro (intervalo de confiança)
    ax.errorbar(
        x=tukey_plot["Diferença média"],
        y=y_pos,
        xerr=[
            tukey_plot["Diferença média"] - tukey_plot["IC 95% inferior"],
            tukey_plot["IC 95% superior"] - tukey_plot["Diferença média"]
        ],
        fmt="o",
        capsize=5,
        linewidth=1.8,
        markersize=5
    )

    # Linha vertical em x=0 como referência
    ax.axvline(
        x=0,
        linestyle="--",
        linewidth=1.2
    )

    # Configuração do eixo y
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tukey_plot["Comparação"])

    # Rótulos e título
    ax.set_title(
        "Intervalos de confiança de 95% - Tukey",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(f"Diferença média - {nome_ensaio} ({unidade})")
    ax.set_ylabel("Comparações entre grupos")

    # Grade para melhor leitura
    ax.grid(True, axis="x", alpha=0.30)
    ax.grid(True, axis="y", alpha=0.15)

    # Inverte eixo Y para que maiores índices apareçam no topo
    ax.invert_yaxis()

    fig.tight_layout()

    return fig


# ============================================================
# FUNÇÕES DE EXPORTAÇÃO
# ============================================================

def gerar_excel(dados_brutos, descritiva, anova_formatada, pressupostos, tukey_df, interpretacao):
    """
    Gera arquivo Excel com todas as tabelas de resultados.
    
    Abas do arquivo:
    - Dados usados: dados brutos após limpeza
    - Descritiva: estatística descritiva por grupo
    - ANOVA: tabela de ANOVA
    - Pressupostos: resultados dos testes de normalidade e homogeneidade
    - Tukey: tabela de comparações múltiplas de Tukey
    - Interpretacao: interpretação automática dos resultados
    
    Parâmetros:
        dados_brutos (pd.DataFrame): Dados após limpeza
        descritiva (pd.DataFrame): Estatística descritiva
        anova_formatada (pd.DataFrame): Tabela ANOVA
        pressupostos (pd.DataFrame): Testes de pressupostos
        tukey_df (pd.DataFrame): Tabela de Tukey
        interpretacao (pd.DataFrame): Interpretação automática
        
    Retorna:
        BytesIO: Buffer com arquivo Excel
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dados_brutos.to_excel(writer, sheet_name="Dados usados", index=False)
        descritiva.to_excel(writer, sheet_name="Descritiva", index=False)
        anova_formatada.to_excel(writer, sheet_name="ANOVA", index=False)
        pressupostos.to_excel(writer, sheet_name="Pressupostos", index=False)
        tukey_df.to_excel(writer, sheet_name="Tukey", index=False)
        interpretacao.to_excel(writer, sheet_name="Interpretacao", index=False)

    output.seek(0)
    return output


def converter_figura_para_png(fig):
    """
    Converte figura matplotlib para PNG com alta resolução.
    
    Parâmetros:
        fig (matplotlib.figure.Figure): Figura a ser convertida
        
    Retorna:
        BytesIO: Buffer com imagem PNG
    """
    output = BytesIO()
    fig.savefig(output, format="png", dpi=DPI_PNG, bbox_inches="tight")
    output.seek(0)
    return output


def converter_figura_para_pdf(fig):
    """
    Converte figura matplotlib para PDF.
    
    Parâmetros:
        fig (matplotlib.figure.Figure): Figura a ser convertida
        
    Retorna:
        BytesIO: Buffer com arquivo PDF
    """
    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    output.seek(0)
    return output


# ============================================================
# FUNÇÕES DE INTERPRETAÇÃO
# ============================================================

def gerar_interpretacao_automatica(p_valor, tukey_df, alpha, nome_ensaio):
    """
    Gera interpretação automática dos resultados da análise.
    
    Parâmetros:
        p_valor (float): P-valor da ANOVA
        tukey_df (pd.DataFrame): Tabela de Tukey
        alpha (float): Nível de significância
        nome_ensaio (str): Nome do ensaio
        
    Retorna:
        pd.DataFrame: Tabela com interpretação
    """
    # Interpretação da ANOVA
    if p_valor < alpha:
        texto_anova = (
            f"A ANOVA indicou diferença estatisticamente significativa "
            f"entre os grupos para {nome_ensaio}, considerando α = {alpha}."
        )
    else:
        texto_anova = (
            f"A ANOVA não indicou diferença estatisticamente significativa "
            f"entre os grupos para {nome_ensaio}, considerando α = {alpha}."
        )

    # Interpretação do Tukey
    comparacoes_significativas = tukey_df[tukey_df["Significativo"] == True]

    if comparacoes_significativas.empty:
        texto_tukey = (
            "O teste post hoc de Tukey não identificou diferenças significativas "
            "entre as comparações par a par."
        )
    else:
        texto_tukey = (
            "O teste post hoc de Tukey identificou diferenças significativas "
            "em pelo menos uma comparação par a par."
        )

    interpretacao = pd.DataFrame({
        "Item": ["ANOVA", "Tukey"],
        "Resultado": [texto_anova, texto_tukey]
    })

    return interpretacao, texto_anova, texto_tukey


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

st.sidebar.header("1. Enviar planilha")

arquivo = st.sidebar.file_uploader(
    "Envie uma planilha Excel ou CSV",
    type=["xlsx", "csv"]
)


if arquivo is None:

    st.info("Envie uma planilha Excel ou CSV para iniciar a análise.")

    st.markdown(
        """
        ### Modelo esperado da planilha

        | Traco | CP | Flexao_MPa |
        |---|---:|---:|
        | T-REF | 4 | 6,26 |
        | T-REF | 5 | 7,01 |
        | T-REF | 6 | 7,67 |
        | T-0,4 | 4 | 8,93 |
        | T-0,4 | 5 | 7,91 |
        | T-0,4 | 6 | 7,21 |

        No app, escolha:

        - **Coluna dos grupos:** `Traco`
        - **Coluna dos valores numéricos:** `Flexao_MPa`
        """
    )

else:

    # ============================================================
    # LEITURA E SELEÇÃO DE ABA
    # ============================================================

    try:
        if arquivo.name.endswith(".csv"):
            df = pd.read_csv(arquivo)
        else:
            df = selecionar_aba_excel(arquivo)

    except Exception as erro:
        st.error("Erro ao ler a planilha.")
        st.exception(erro)
        st.stop()

    st.subheader("Pré-visualização da planilha")
    st.dataframe(df)

    # ============================================================
    # CONFIGURAÇÕES DA ANÁLISE
    # ============================================================

    st.sidebar.header("2. Configurar análise")

    coluna_grupo = st.sidebar.selectbox(
        "Coluna dos grupos",
        df.columns
    )

    coluna_resposta = st.sidebar.selectbox(
        "Coluna dos valores numéricos",
        df.columns
    )

    nome_ensaio = st.sidebar.text_input(
        "Nome do ensaio",
        value="Resistência à tração na flexão aos 28 dias"
    )

    unidade = st.sidebar.text_input(
        "Unidade",
        value="MPa"
    )

    alpha = st.sidebar.selectbox(
        "Nível de significância",
        [0.05, 0.01, 0.10],
        index=0
    )

    ordem_grupos_texto = st.sidebar.text_input(
        "Ordem dos grupos para o Tukey",
        value="T-REF, T-0,4, T-0,6, T-0,8, T-1,0"
    )

    ordem_grupos = [
        grupo.strip()
        for grupo in ordem_grupos_texto.split(",")
        if grupo.strip() != ""
    ]

    rodar = st.sidebar.button("Rodar análise")

    if rodar:

        try:
            # ============================================================
            # LIMPEZA DOS DADOS
            # ============================================================

            dados = limpar_dados(df, coluna_grupo, coluna_resposta)

            # Validações básicas
            if dados.empty:
                st.error(
                    "Depois da limpeza, não sobraram dados válidos. "
                    "Confira se a coluna dos grupos e a coluna numérica foram selecionadas corretamente."
                )
                st.stop()

            if dados["Grupo"].nunique() < 2:
                st.error(
                    "A ANOVA precisa de pelo menos dois grupos diferentes. "
                    "Confira se a coluna dos grupos foi selecionada corretamente."
                )
                st.stop()

            # Verifica quantidade de replicatas
            contagem_grupos = dados.groupby("Grupo")["Resultado"].count()

            if (contagem_grupos < 2).any():
                st.warning(
                    "Atenção: pelo menos um grupo possui menos de 2 valores. "
                    "O ideal para ANOVA/Tukey é ter replicatas por grupo."
                )

            # Verifica grupos que não estão na ordem definida
            grupos_na_planilha = list(dados["Grupo"].unique())
            grupos_fora_da_ordem = [
                grupo for grupo in grupos_na_planilha
                if grupo not in ordem_grupos
            ]

            if len(grupos_fora_da_ordem) > 0:
                st.warning(
                    "Atenção: alguns grupos da planilha não estão na ordem definida para o Tukey: "
                    + ", ".join(grupos_fora_da_ordem)
                    + ". Eles serão mantidos, mas a ordenação pode não ficar igual ao MATLAB."
                )

            # Exibe dados após limpeza
            st.subheader("Dados usados na análise após limpeza")
            st.dataframe(dados)

            st.subheader("Número de valores por grupo")
            st.dataframe(contagem_grupos.reset_index(name="N"))

            # ============================================================
            # CÁLCULOS ESTATÍSTICOS
            # ============================================================

            # Estatística descritiva
            descritiva = calcular_estatistica_descritiva(dados)

            # ANOVA
            modelo, anova_formatada, (f_valor, p_valor) = executar_anova(dados)

            # Pressupostos
            pressupostos = avaliar_pressupostos(modelo, dados, alpha)

            # Teste de Tukey
            tukey_df = executar_tukey(dados, alpha, ordem_grupos)

            # Interpretação automática
            interpretacao, texto_anova, texto_tukey = gerar_interpretacao_automatica(
                p_valor, tukey_df, alpha, nome_ensaio
            )

            # ============================================================
            # EXIBIÇÃO DOS RESULTADOS
            # ============================================================

            st.success("Análise concluída com sucesso!")

            st.subheader("Estatística descritiva")
            st.dataframe(descritiva)

            st.subheader("Tabela de ANOVA")
            st.dataframe(anova_formatada)

            st.subheader("Testes de pressupostos")
            st.dataframe(pressupostos)

            st.subheader("Tabela de Tukey")
            st.dataframe(tukey_df)

            st.subheader("Gráfico de Tukey")
            fig = gerar_grafico_tukey(tukey_df, nome_ensaio, unidade)
            st.pyplot(fig)

            st.subheader("Interpretação automática")
            st.write(texto_anova)
            st.write(texto_tukey)

            # ============================================================
            # DOWNLOADS
            # ============================================================

            excel_file = gerar_excel(
                dados,
                descritiva,
                anova_formatada,
                pressupostos,
                tukey_df,
                interpretacao
            )

            png_file = converter_figura_para_png(fig)
            pdf_file = converter_figura_para_pdf(fig)

            st.divider()
            st.subheader("Baixar resultados")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="📥 Excel",
                    data=excel_file,
                    file_name="resultado_anova_tukey.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col2:
                st.download_button(
                    label="📥 PNG (600 dpi)",
                    data=png_file,
                    file_name="grafico_tukey.png",
                    mime="image/png"
                )

            with col3:
                st.download_button(
                    label="📥 PDF",
                    data=pdf_file,
                    file_name="grafico_tukey.pdf",
                    mime="application/pdf"
                )

        except Exception as erro:
            st.error("Ocorreu um erro durante a análise.")
            st.exception(erro)