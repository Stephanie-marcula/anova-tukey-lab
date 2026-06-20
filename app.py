import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from scipy.stats import shapiro, levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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


PRECISAO_SAIDA = 4
DPI_PNG = 600
TAMANHO_FIGURA = (9, 5.5)
FIGSIZE_WIDTH = 9
FIGSIZE_HEIGHT = 5.5


def limpar_dados(df, coluna_grupo, coluna_resposta):

    dados = df[[coluna_grupo, coluna_resposta]].copy()
    dados.columns = ["Grupo", "Resultado"]


    dados["Grupo"] = dados["Grupo"].astype(str).str.strip()


    dados["Resultado"] = (
        dados["Resultado"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )


    dados["Resultado"] = pd.to_numeric(dados["Resultado"], errors="coerce")


    dados = dados.dropna(subset=["Grupo", "Resultado"])


    dados = dados[
        (dados["Grupo"] != "") &
        (dados["Grupo"].str.lower() != "nan")
    ]

    return dados


def ler_planilha(arquivo):


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


    xls = pd.ExcelFile(arquivo)
    aba = st.sidebar.selectbox("Escolha a aba da planilha", xls.sheet_names)
    df = pd.read_excel(arquivo, sheet_name=aba)
    return df


def calcular_estatistica_descritiva(dados):


    descritiva = dados.groupby("Grupo")["Resultado"].agg(
        N="count",
        Média="mean",
        Desvio_Padrão="std",
        Erro_Padrão=lambda x: x.std() / np.sqrt(len(x))
    ).reset_index()


    for col in ["Média", "Desvio_Padrão", "Erro_Padrão"]:
        descritiva[col] = descritiva[col].round(PRECISAO_SAIDA)

    return descritiva


def executar_anova(dados):


    modelo = ols("Resultado ~ C(Grupo)", data=dados).fit()
    anova = sm.stats.anova_lm(modelo, typ=2)


    sq_fator = anova.loc["C(Grupo)", "sum_sq"]
    gl_fator = anova.loc["C(Grupo)", "df"]
    mq_fator = sq_fator / gl_fator
    f_valor = anova.loc["C(Grupo)", "F"]
    p_valor = anova.loc["C(Grupo)", "PR(>F)"]

    sq_residuos = anova.loc["Residual", "sum_sq"]
    gl_residuos = anova.loc["Residual", "df"]
    mq_residuos = sq_residuos / gl_residuos


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


    residuos = modelo.resid

    if len(residuos) >= 3:
        shapiro_stat, shapiro_p = shapiro(residuos)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan


    grupos = [
        grupo["Resultado"].values
        for nome, grupo in dados.groupby("Grupo")
    ]
    levene_stat, levene_p = levene(*grupos)


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


        if g1_original in ordem and g2_original in ordem:
            idx_g1 = ordem[g1_original]
            idx_g2 = ordem[g2_original]


            if idx_g1 > idx_g2:

                grupo_maior = g1_original
                grupo_menor = g2_original
                diff_final = diff_original
                lower_final = lower_original
                upper_final = upper_original
                idx_maior = idx_g1
                idx_menor = idx_g2
            else:


                grupo_maior = g2_original
                grupo_menor = g1_original
                diff_final = -diff_original
                lower_final = -upper_original
                upper_final = -lower_original
                idx_maior = idx_g2
                idx_menor = idx_g1

        else:

            grupo_maior = g1_original
            grupo_menor = g2_original
            diff_final = diff_original
            lower_final = lower_original
            upper_final = upper_original
            idx_maior = 999
            idx_menor = 999


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


    tukey_padronizado = tukey_padronizado.sort_values(
        by=["idx_maior", "idx_menor"],
        ascending=[False, False]
    ).reset_index(drop=True)


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


    tukey_padronizado["Diferença média"] = tukey_padronizado["Diferença média"].round(PRECISAO_SAIDA)
    tukey_padronizado["IC 95% inferior"] = tukey_padronizado["IC 95% inferior"].round(PRECISAO_SAIDA)
    tukey_padronizado["IC 95% superior"] = tukey_padronizado["IC 95% superior"].round(PRECISAO_SAIDA)
    tukey_padronizado["p-ajustado"] = tukey_padronizado["p-ajustado"].round(PRECISAO_SAIDA)

    return tukey_padronizado


def executar_tukey(dados, alpha, ordem_grupos):


    tukey = pairwise_tukeyhsd(
        endog=dados["Resultado"],
        groups=dados["Grupo"],
        alpha=alpha
    )


    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )


    tukey_df = tukey_df.rename(columns={
        "group1": "Grupo 1",
        "group2": "Grupo 2",
        "meandiff": "Diferença média",
        "p-adj": "p-ajustado",
        "lower": "IC 95% inferior",
        "upper": "IC 95% superior",
        "reject": "Significativo"
    })


    tukey_df = padronizar_tukey_por_ordem(tukey_df, ordem_grupos)

    return tukey_df


def gerar_grafico_tukey(tukey_df, nome_ensaio, unidade):


    tukey_plot = tukey_df.copy().reset_index(drop=True)


    y_pos = np.arange(len(tukey_plot))

    fig, ax = plt.subplots(figsize=(FIGSIZE_WIDTH, FIGSIZE_HEIGHT))


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


    ax.axvline(
        x=0,
        linestyle="--",
        linewidth=1.2
    )


    ax.set_yticks(y_pos)
    ax.set_yticklabels(tukey_plot["Comparação"])


    ax.set_title(
        "Intervalos de confiança de 95% - Tukey",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(f"Diferença média - {nome_ensaio} ({unidade})")
    ax.set_ylabel("Comparações entre grupos")


    ax.grid(True, axis="x", alpha=0.30)
    ax.grid(True, axis="y", alpha=0.15)


    ax.invert_yaxis()

    fig.tight_layout()

    return fig


def gerar_excel(dados_brutos, descritiva, anova_formatada, pressupostos, tukey_df, interpretacao):


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


    output = BytesIO()
    fig.savefig(output, format="png", dpi=DPI_PNG, bbox_inches="tight")
    output.seek(0)
    return output


def converter_figura_para_pdf(fig):


    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    output.seek(0)
    return output


def gerar_interpretacao_automatica(p_valor, tukey_df, alpha, nome_ensaio):


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


            dados = limpar_dados(df, coluna_grupo, coluna_resposta)


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


            contagem_grupos = dados.groupby("Grupo")["Resultado"].count()

            if (contagem_grupos < 2).any():
                st.warning(
                    "Atenção: pelo menos um grupo possui menos de 2 valores. "
                    "O ideal para ANOVA/Tukey é ter replicatas por grupo."
                )


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


            st.subheader("Dados usados na análise após limpeza")
            st.dataframe(dados)

            st.subheader("Número de valores por grupo")
            st.dataframe(contagem_grupos.reset_index(name="N"))


            descritiva = calcular_estatistica_descritiva(dados)


            modelo, anova_formatada, (f_valor, p_valor) = executar_anova(dados)


            pressupostos = avaliar_pressupostos(modelo, dados, alpha)


            tukey_df = executar_tukey(dados, alpha, ordem_grupos)


            interpretacao, texto_anova, texto_tukey = gerar_interpretacao_automatica(
                p_valor, tukey_df, alpha, nome_ensaio
            )


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
