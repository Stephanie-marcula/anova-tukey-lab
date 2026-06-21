import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill

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
    - Exportação dos resultados em Excel, PDF e PNG.
    """
)


PRECISAO_SAIDA = 4
PRECISAO_ANOVA = 5
DPI_PNG = 600
FIGSIZE_WIDTH = 9
FIGSIZE_HEIGHT = 5.5


def formatar_p_valor(p):
    if pd.isna(p):
        return ""

    if p == 0:
        return "< 0.001"

    if p < 0.001:
        return f"{p:.2E}"

    return round(p, PRECISAO_SAIDA)


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
        Erro_Padrão=lambda x: x.std() / np.sqrt(len(x)),
        Mínimo="min",
        Máximo="max"
    ).reset_index()

    descritiva["CV (%)"] = (
        descritiva["Desvio_Padrão"] / descritiva["Média"] * 100
    )

    colunas_numericas = [
        "Média",
        "Desvio_Padrão",
        "Erro_Padrão",
        "Mínimo",
        "Máximo",
        "CV (%)"
    ]

    for coluna in colunas_numericas:
        descritiva[coluna] = descritiva[coluna].round(PRECISAO_SAIDA)

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
        "SQ": [
            f"{sq_fator:.5f}",
            f"{sq_residuos:.5f}"
        ],
        "MQ": [
            f"{mq_fator:.5f}",
            f"{mq_residuos:.5f}"
        ],
        "F": [
            f"{f_valor:.5f}",
            ""
        ],
        "P. valor": [
            formatar_p_valor(p_valor),
            ""
        ]
    })

    return modelo, anova_formatada, f_valor, p_valor


def avaliar_pressupostos(modelo, dados, alpha):
    residuos = modelo.resid

    if len(residuos) >= 3:
        shapiro_stat, shapiro_p = shapiro(residuos)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    grupos = [
        grupo["Resultado"].values
        for _, grupo in dados.groupby("Grupo")
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
            formatar_p_valor(shapiro_p) if not np.isnan(shapiro_p) else "",
            formatar_p_valor(levene_p)
        ],
        "Interpretação": [
            interpretacao_shapiro,
            interpretacao_levene
        ]
    })

    return pressupostos


def inverter_comparacao(diff, lower, upper):
    diff_final = -diff
    lower_final = -upper
    upper_final = -lower
    return diff_final, lower_final, upper_final


def padronizar_tukey_por_referencia(tukey_df, ordem_grupos, grupo_referencia):
    ordem = {grupo: i for i, grupo in enumerate(ordem_grupos)}
    linhas = []

    for _, row in tukey_df.iterrows():
        g1 = str(row["Grupo 1"]).strip()
        g2 = str(row["Grupo 2"]).strip()

        diff = float(row["Diferença média"])
        lower = float(row["IC 95% inferior"])
        upper = float(row["IC 95% superior"])

        pvalor = row["p-ajustado"]
        significativo = row["Significativo"]

        idx_g1 = ordem.get(g1, 999)
        idx_g2 = ordem.get(g2, 999)

        if g1 == grupo_referencia and g2 != grupo_referencia:
            grupo_base = g1
            grupo_comparado = g2
            diff_final = diff
            lower_final = lower
            upper_final = upper
            prioridade = 0
            ordem_base = idx_g2

        elif g2 == grupo_referencia and g1 != grupo_referencia:
            grupo_base = g2
            grupo_comparado = g1
            diff_final, lower_final, upper_final = inverter_comparacao(
                diff,
                lower,
                upper
            )
            prioridade = 0
            ordem_base = idx_g1

        else:
            if idx_g2 > idx_g1:
                grupo_base = g1
                grupo_comparado = g2
                diff_final = diff
                lower_final = lower
                upper_final = upper
                ordem_base = idx_g2
            else:
                grupo_base = g2
                grupo_comparado = g1
                diff_final, lower_final, upper_final = inverter_comparacao(
                    diff,
                    lower,
                    upper
                )
                ordem_base = idx_g1

            prioridade = 1

        comparacao = f"{grupo_comparado} - {grupo_base}"

        linhas.append({
            "Comparação": comparacao,
            "Grupo 1": grupo_base,
            "Grupo 2": grupo_comparado,
            "Diferença média": diff_final,
            "IC 95% inferior": lower_final,
            "IC 95% superior": upper_final,
            "p-ajustado": pvalor,
            "Significativo": significativo,
            "Prioridade": prioridade,
            "Ordem": ordem_base
        })

    tukey_padronizado = pd.DataFrame(linhas)

    tukey_padronizado = tukey_padronizado.sort_values(
        by=["Prioridade", "Ordem"],
        ascending=[True, True]
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

    tukey_padronizado["Diferença média"] = (
        tukey_padronizado["Diferença média"].round(PRECISAO_SAIDA)
    )

    tukey_padronizado["IC 95% inferior"] = (
        tukey_padronizado["IC 95% inferior"].round(PRECISAO_SAIDA)
    )

    tukey_padronizado["IC 95% superior"] = (
        tukey_padronizado["IC 95% superior"].round(PRECISAO_SAIDA)
    )

    tukey_padronizado["p-ajustado"] = (
        tukey_padronizado["p-ajustado"].apply(formatar_p_valor)
    )

    return tukey_padronizado


def executar_tukey(dados, alpha, ordem_grupos, grupo_referencia):
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

    tukey_df = padronizar_tukey_por_referencia(
        tukey_df,
        ordem_grupos,
        grupo_referencia
    )

    return tukey_df


def gerar_grafico_tukey(tukey_df, nome_ensaio, unidade, grupo_referencia):
    tukey_plot = tukey_df.copy().reset_index(drop=True)

    for col in ["Diferença média", "IC 95% inferior", "IC 95% superior"]:
        tukey_plot[col] = pd.to_numeric(tukey_plot[col], errors="coerce")

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
        ecolor="steelblue",
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        capsize=5,
        linewidth=1.4,
        markersize=4
    )

    ax.axvline(
        x=0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75
    )

    n_ref = tukey_plot["Comparação"].str.contains(grupo_referencia, regex=False).sum()

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

    fig.tight_layout()

    return fig


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


def escrever_tabela_excel(worksheet, workbook, titulo, df_tabela, linha_inicial):
    formato_secao = workbook.add_format({
        "bold": True,
        "font_size": 12,
        "bg_color": "#D9EAF7",
        "border": 1
    })

    formato_cabecalho = workbook.add_format({
        "bold": True,
        "bg_color": "#EDEDED",
        "border": 1,
        "align": "center",
        "valign": "vcenter"
    })

    formato_texto = workbook.add_format({
        "border": 1,
        "valign": "top"
    })

    formato_numero = workbook.add_format({
        "border": 1,
        "num_format": "0.00000",
        "valign": "top"
    })

    ultima_coluna = max(len(df_tabela.columns) - 1, 1)

    worksheet.merge_range(
        linha_inicial,
        0,
        linha_inicial,
        ultima_coluna,
        titulo,
        formato_secao
    )

    linha_tabela = linha_inicial + 1

    for col_idx, col_nome in enumerate(df_tabela.columns):
        worksheet.write(linha_tabela, col_idx, col_nome, formato_cabecalho)

    for row_idx, (_, row) in enumerate(df_tabela.iterrows(), start=linha_tabela + 1):
        for col_idx, valor in enumerate(row):
            if isinstance(valor, (int, float, np.integer, np.floating)) and not pd.isna(valor):
                worksheet.write(row_idx, col_idx, valor, formato_numero)
            else:
                worksheet.write(row_idx, col_idx, valor, formato_texto)

    return linha_tabela + len(df_tabela) + 3


def gerar_excel_relatorio(
    dados,
    descritiva,
    anova_formatada,
    pressupostos,
    tukey_df,
    interpretacao,
    fig,
    nome_ensaio
):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        sheet_relatorio = "Relatorio"
        worksheet = workbook.add_worksheet(sheet_relatorio)
        writer.sheets[sheet_relatorio] = worksheet

        formato_titulo = workbook.add_format({
            "bold": True,
            "font_size": 16,
            "align": "center",
            "valign": "vcenter"
        })

        formato_obs = workbook.add_format({
            "italic": True,
            "font_size": 10
        })

        worksheet.set_column("A:A", 24)
        worksheet.set_column("B:B", 20)
        worksheet.set_column("C:C", 20)
        worksheet.set_column("D:D", 20)
        worksheet.set_column("E:E", 20)
        worksheet.set_column("F:F", 20)
        worksheet.set_column("G:G", 20)
        worksheet.set_column("H:H", 20)

        linha = 0

        worksheet.merge_range(
            linha,
            0,
            linha,
            7,
            f"Relatório estatístico - {nome_ensaio}",
            formato_titulo
        )

        linha += 2

        linha = escrever_tabela_excel(
            worksheet,
            workbook,
            "1. Dados usados na análise",
            dados,
            linha
        )

        linha = escrever_tabela_excel(
            worksheet,
            workbook,
            "2. Estatística descritiva",
            descritiva,
            linha
        )

        linha = escrever_tabela_excel(
            worksheet,
            workbook,
            "3. Tabela da ANOVA",
            anova_formatada,
            linha
        )

        worksheet.write(linha, 0, "Onde:", formato_obs)
        worksheet.write(
            linha + 1,
            0,
            "G.L. = graus de liberdade; SQ = soma dos quadrados; MQ = média quadrática; F = razão entre as médias quadráticas.",
            formato_obs
        )

        linha += 4

        linha = escrever_tabela_excel(
            worksheet,
            workbook,
            "4. Testes de pressupostos",
            pressupostos,
            linha
        )

        linha = escrever_tabela_excel(
            worksheet,
            workbook,
            "5. Teste post hoc de Tukey",
            tukey_df,
            linha
        )

        linha = escrever_tabela_excel(
            worksheet,
            workbook,
            "6. Interpretação automática",
            interpretacao,
            linha
        )

        sheet_grafico = "Grafico Tukey"
        worksheet_grafico = workbook.add_worksheet(sheet_grafico)
        writer.sheets[sheet_grafico] = worksheet_grafico

        formato_titulo_grafico = workbook.add_format({
            "bold": True,
            "font_size": 14,
            "align": "center",
            "valign": "vcenter"
        })

        worksheet_grafico.set_column("A:A", 20)
        worksheet_grafico.set_column("B:B", 20)
        worksheet_grafico.set_column("C:C", 20)
        worksheet_grafico.set_column("D:D", 20)
        worksheet_grafico.set_column("E:E", 20)

        worksheet_grafico.merge_range(
            0,
            0,
            0,
            5,
            f"Gráfico de Tukey - {nome_ensaio}",
            formato_titulo_grafico
        )

        imagem_grafico = BytesIO()
        fig.savefig(
            imagem_grafico,
            format="png",
            dpi=DPI_PNG,
            bbox_inches="tight"
        )
        imagem_grafico.seek(0)

        worksheet_grafico.insert_image(
            2,
            0,
            "grafico_tukey.png",
            {
                "image_data": imagem_grafico,
                "x_scale": 0.95,
                "y_scale": 0.95
            }
        )

    output.seek(0)
    return output


def adicionar_tabela_pdf(pdf, titulo, df_tabela):
    df_texto = df_tabela.copy()

    for col in df_texto.columns:
        df_texto[col] = df_texto[col].astype(str).apply(lambda x: fill(x, width=35))

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        titulo,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes
    )

    tabela = ax.table(
        cellText=df_texto.values,
        colLabels=df_texto.columns,
        loc="center",
        cellLoc="center"
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8)
    tabela.scale(1, 1.4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def adicionar_texto_pdf(pdf, titulo, textos):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        titulo,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes
    )

    y = 0.82

    for texto in textos:
        ax.text(
            0.08,
            y,
            fill(texto, width=120),
            ha="left",
            va="top",
            fontsize=11,
            transform=ax.transAxes
        )
        y -= 0.12

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def gerar_pdf_relatorio(
    dados,
    descritiva,
    anova_formatada,
    pressupostos,
    tukey_df,
    interpretacao,
    fig_grafico,
    nome_ensaio
):
    output = BytesIO()

    with PdfPages(output) as pdf:
        adicionar_texto_pdf(
            pdf,
            f"Relatório estatístico - {nome_ensaio}",
            [
                "Relatório gerado automaticamente pelo aplicativo ANOVA e Tukey Lab.",
                "A análise inclui estatística descritiva, ANOVA de uma via, testes de pressupostos, teste post hoc de Tukey e gráfico dos intervalos de confiança de 95%."
            ]
        )

        adicionar_tabela_pdf(pdf, "1. Dados usados na análise", dados)
        adicionar_tabela_pdf(pdf, "2. Estatística descritiva", descritiva)
        adicionar_tabela_pdf(pdf, "3. Tabela da ANOVA", anova_formatada)

        adicionar_texto_pdf(
            pdf,
            "Observação sobre a ANOVA",
            [
                "G.L. = graus de liberdade; SQ = soma dos quadrados; MQ = média quadrática; F = razão entre as médias quadráticas."
            ]
        )

        adicionar_tabela_pdf(pdf, "4. Testes de pressupostos", pressupostos)
        adicionar_tabela_pdf(pdf, "5. Teste post hoc de Tukey", tukey_df)
        adicionar_tabela_pdf(pdf, "6. Interpretação automática", interpretacao)

        pdf.savefig(fig_grafico, bbox_inches="tight")

    output.seek(0)
    return output


def converter_figura_para_png(fig):
    output = BytesIO()
    fig.savefig(output, format="png", dpi=DPI_PNG, bbox_inches="tight")
    output.seek(0)
    return output


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

        | Traco | CP | Resultado |
        |---|---:|---:|
        | T-REF | 1 | 29,72191 |
        | T-REF | 2 | 29,50426 |
        | T-REF | 3 | 28,99514 |
        | T-0,4 | 1 | 26,35254 |
        | T-0,4 | 2 | 24,22243 |
        | T-0,4 | 3 | 25,86113 |

        No app, escolha:

        - **Coluna dos grupos:** `Traco`
        - **Coluna dos valores numéricos:** a coluna do ensaio, por exemplo `Modulo_MPa`
        """
    )

else:

    try:
        if arquivo.name.endswith(".csv"):
            df = pd.read_csv(arquivo, sep=None, engine="python")
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
        value="Módulo de elasticidade aos 28 dias"
    )

    unidade = st.sidebar.text_input(
        "Unidade",
        value="GPa"
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

    if len(ordem_grupos) == 0:
        st.sidebar.error("Informe pelo menos um grupo na ordem dos grupos.")
        st.stop()

    grupo_referencia = ordem_grupos[0]

    st.sidebar.caption(
        f"Grupo de referência automático no Tukey: {grupo_referencia}"
    )

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
                    + ". Eles serão mantidos, mas a ordenação pode não ficar igual ao esperado."
                )

            if grupo_referencia not in dados["Grupo"].unique():
                st.warning(
                    f"O grupo de referência automático '{grupo_referencia}' não foi encontrado nos dados. "
                    "Confira a grafia da ordem dos grupos e da coluna de grupos."
                )

            st.subheader("Dados usados na análise após limpeza")
            st.dataframe(dados)

            st.subheader("Número de valores por grupo")
            st.dataframe(contagem_grupos.reset_index(name="N"))

            descritiva = calcular_estatistica_descritiva(dados)

            modelo, anova_formatada, f_valor, p_valor = executar_anova(dados)

            pressupostos = avaliar_pressupostos(modelo, dados, alpha)

            tukey_df = executar_tukey(
                dados,
                alpha,
                ordem_grupos,
                grupo_referencia
            )

            interpretacao, texto_anova, texto_tukey = gerar_interpretacao_automatica(
                p_valor,
                tukey_df,
                alpha,
                nome_ensaio
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
            fig = gerar_grafico_tukey(
                tukey_df,
                nome_ensaio,
                unidade,
                grupo_referencia
            )
            st.pyplot(fig)

            st.subheader("Interpretação automática")
            st.write(texto_anova)
            st.write(texto_tukey)

            excel_file = gerar_excel_relatorio(
                dados,
                descritiva,
                anova_formatada,
                pressupostos,
                tukey_df,
                interpretacao,
                fig,
                nome_ensaio
            )

            pdf_file = gerar_pdf_relatorio(
                dados,
                descritiva,
                anova_formatada,
                pressupostos,
                tukey_df,
                interpretacao,
                fig,
                nome_ensaio
            )

            png_file = converter_figura_para_png(fig)

            st.divider()
            st.subheader("Baixar resultados")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="📥 Excel completo",
                    data=excel_file,
                    file_name="relatorio_anova_tukey.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col2:
                st.download_button(
                    label="📄 PDF completo",
                    data=pdf_file,
                    file_name="relatorio_anova_tukey.pdf",
                    mime="application/pdf"
                )

            with col3:
                st.download_button(
                    label="🖼️ Gráfico PNG",
                    data=png_file,
                    file_name="grafico_tukey.png",
                    mime="image/png"
                )

        except Exception as erro:
            st.error("Ocorreu um erro durante a análise.")
            st.exception(erro)