"""Gera a Ficha de Avaliação Técnica (8 a 16 anos) em Excel.

Uso: python scripts/build_ficha_xlsx.py <site.html> <saida.xlsx>
As normas são lidas do bloco `const TESTS=[...]` da Plataforma de Avaliação Quantitativa ENB
(página HTML guardada localmente), para não haver erros de transcrição.
"""
import re
import sys

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.formatting.rule import ColorScaleRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

NAVY, TEAL, SAND, GREY = "123B5D", "176B87", "FFF2CC", "F2F5F8"
N_ROWS = 25                     # linhas de atletas
FIRST = 5                       # primeira linha de atletas
LAST = FIRST + N_ROWS - 1
thin = Side(style="thin", color="BFC9D2")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)


# ── Normas ──────────────────────────────────────────────────────────────────
def load_site_norms(path):
    html = open(path, encoding="utf-8").read()
    html = re.sub(r"[A-Za-z0-9+/=]{300,}", "B64", html)
    start = html.index("const TESTS=[")
    end = html.index("];", start)
    block = html[start:end]
    out = {}
    for part in re.split(r'\{id:"', block)[1:]:
        tid = part[:2]
        lower = "lower:true" in part
        ref = part[part.index("ref:{"):]
        rows = {int(a): [float(x) for x in v.split(",")]
                for a, v in re.findall(r"(\d+):\[([\d.,]+)\]", ref)}
        out[tid] = {"lower": lower, "ref": rows}
    return out


def build_blocks(site):
    """[(teste, idade, [(percentil, publicado), ...] por resultado crescente)]"""
    spec = [("Drible", "t2", range(8, 17), "Matulaitis et al. (2019)"),
            ("Defesa", "t1", range(8, 17), "Matulaitis et al. (2019)"),
            ("Pontos Marcados", "t3", range(8, 11), "Matulaitis et al. (2019)"),
            ("Lançamento 1 Minuto", "t5", range(11, 17), "One Minute Shooting, Tabela S7 (masc.)"),
            ("Illinois com Drible", "t4", range(13, 17), "Getchell et al. (1998), Sabonis tab. 3.20"),
            ("Velocidade e Coordenação", "t6", range(13, 17), "T-Test adaptado (Sabonis)"),
            ("Lances Livres", "t7", range(13, 17), "Stonkus (2002), Tabela S6 (masc.)")]
    pct = [90, 80, 70, 60, 50, 40, 30, 20, 5]        # P10 («<10») registado como 5, como na ficha original
    blocks, lower = [], {}
    for name, tid, ages, src in spec:
        lower[name] = site[tid]["lower"]
        for age in ages:
            vals = site[tid]["ref"][age]
            pairs = list(zip(pct, vals))
            if not site[tid]["lower"]:
                pairs.reverse()                     # resultado crescente ↔ percentil crescente
            blocks.append((name, age, pairs, src))
    return blocks, lower


# ── Estilos ─────────────────────────────────────────────────────────────────
def title(ws, text, sub, last_col):
    ws.merge_cells(f"A1:{last_col}1")
    ws.merge_cells(f"A2:{last_col}2")
    ws["A1"], ws["A2"] = text, sub
    ws["A1"].font = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws["A2"].font = Font(italic=True, size=10, color="243746")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height, ws.row_dimensions[2].height = 30, 32


def header(ws, row, labels, fill=TEAL):
    for i, lab in enumerate(labels, 1):
        c = ws.cell(row, i, lab)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=fill)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BOX
    ws.row_dimensions[row].height = 34


# ── Fórmulas de percentil (genéricas: um bloco de 9 linhas na folha Normas) ──
def pct_formulas(row, val, start, idx):
    G, D = "Normas!$F:$F", "Normas!$D:$D"
    s, i = f"{start}{row}", f"{idx}{row}"
    v = f"{val}{row}"
    start_formula = None  # preenchido pelo chamador
    interp = (f"INDEX({D},{s}+{i}-1)+({v}-INDEX({G},{s}+{i}-1))*(INDEX({D},{s}+{i})-INDEX({D},{s}+{i}-1))"
              f"/(INDEX({G},{s}+{i})-INDEX({G},{s}+{i}-1))")
    main = (f'=IF(OR($D{row}="",{v}=""),"",IF({s}="","Sem norma",IF({v}<=INDEX({G},{s}),INDEX({D},{s}),'
            f'IF({v}>=INDEX({G},{s}+8),INDEX({D},{s}+8),IFERROR({interp},"")))))')
    idx_f = (f'=IF(OR({s}="",{v}=""),"",IFERROR(MATCH({v},INDEX({G},{s}):INDEX({G},{s}+8),1),""))')
    return main, idx_f


def key_formula(test_expr, row):
    return f'=IF($D{row}="","",IFERROR(MATCH({test_expr}&"|"&$D{row},Normas!$A:$A,0),""))'


def make_register(ws, kind, example=None):
    """kind 'main' (8–16: Drible, Defesa, Lançamento) ou 'bat' (13–16: bateria IV–VII + Lançamento)."""
    from openpyxl.utils import get_column_letter as L
    if kind == "main":
        tests = [("Drible (s)", "Drible", 5), ("Defesa (s)", "Defesa", 7),
                 ("Lançamento (pontos)", 'IF($D{r}<=10,"Pontos Marcados","Lançamento 1 Minuto")', 9)]
        age_lo, age_hi = 8, 16
        heads = ["N.º", "Nome", "Clube", "Idade (anos)", "Drible (s)", "Percentil Drible", "Defesa (s)",
                 "Percentil Defesa", "Lançamento (pontos)", "Percentil Lançamento", "Observações"]
        lower = [True, True, False]
        sub = ("Introduza idade (8–16) e o melhor resultado de cada teste. Os percentis calculam-se "
               "automaticamente por interpolação entre os valores de referência da folha Normas.")
        ttl = "AVALIAÇÃO TÉCNICA — DRIBLE | DEFESA | LANÇAMENTO  (8 A 16 ANOS)"
    else:
        tests = [("IV. Illinois com Drible (s)", "Illinois com Drible", 5),
                 ("V. Lançamento 1 Minuto (pts)", "Lançamento 1 Minuto", 7),
                 ("VI. Velocidade e Coordenação (s)", "Velocidade e Coordenação", 9),
                 ("VII. Lances Livres (em 30)", "Lances Livres", 11)]
        age_lo, age_hi = 13, 16
        heads = ["N.º", "Nome", "Clube", "Idade (anos)", "IV. Illinois com Drible (s)", "Percentil Illinois",
                 "V. Lançamento 1 Minuto (pts)", "Percentil Lançamento", "VI. Velocidade e Coordenação (s)",
                 "Percentil Vel./Coord.", "VII. Lances Livres (em 30)", "Percentil Lances Livres", "Observações"]
        lower = [True, False, True, False]
        sub = ("Introduza idade (13–16) e o melhor resultado de cada teste. Os percentis calculam-se "
               "automaticamente por interpolação entre os valores de referência da folha Normas.")
        ttl = "AVALIAÇÃO TÉCNICA — BATERIA IV A VII  (13 A 16 ANOS)"
    ncol = len(heads)
    obs_col = ncol
    helper0 = ncol + 2                          # coluna A de ajudantes (oculta), deixa 1 em branco
    title(ws, ttl if not example else "EXEMPLO — " + ttl, sub if not example else
          "Dados fictícios só para demonstrar o preenchimento e o cálculo. Não usar como registo.", L(ncol))
    header(ws, 4, heads)
    ws.freeze_panes = "C5"
    widths = [6, 24, 18, 11] + [14, 13] * len(tests) + [26]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[L(i)].width = w
    for r in range(FIRST, LAST + 1):
        ws.cell(r, 1, r - FIRST + 1)
        for c in range(1, ncol + 1):
            cell = ws.cell(r, c)
            cell.border = BOX
            cell.alignment = Alignment(horizontal="left" if c in (2, 3, obs_col) else "center", vertical="center")
            cell.fill = PatternFill("solid", fgColor="FFFFFF" if r % 2 else GREY)
        for t, (label, test_expr, vcol) in enumerate(tests):
            vcl = L(vcol)
            scol, icol = L(helper0 + 2 * t), L(helper0 + 2 * t + 1)
            ws[f"{scol}{r}"] = key_formula(test_expr.format(r=r) if "{r}" in test_expr else f'"{test_expr}"', r)
            main, idx = pct_formulas(r, vcl, scol, icol)
            ws[f"{icol}{r}"] = idx
            ws.cell(r, vcol + 1, main).number_format = "0"
            ws.cell(r, vcol).number_format = "0.00" if lower[t] else "0"
    # estatísticas
    stats = [("MÉDIA", "AVERAGE"), ("MELHOR RESULTADO", None), ("PIOR RESULTADO", None)]
    for k, (lab, fn) in enumerate(stats):
        r = LAST + 1 + k
        ws.cell(r, 1, lab)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=4)
        for c in range(1, ncol + 1):
            ws.cell(r, c).fill = PatternFill("solid", fgColor=SAND)
            ws.cell(r, c).font = Font(bold=True, color="243746")
            ws.cell(r, c).border = BOX
            ws.cell(r, c).alignment = Alignment(horizontal="center")
        for t in range(len(tests)):
            for off, is_pct in ((0, False), (1, True)):
                col = tests[t][2] + off
                rng = f"{L(col)}{FIRST}:{L(col)}{LAST}"
                better_low = lower[t] and not is_pct
                if fn:
                    f = f'=IFERROR(AVERAGE({rng}),"")'
                elif k == 1:
                    f = f'=IFERROR({"MIN" if better_low else "MAX"}({rng}),"")'
                else:
                    f = f'=IFERROR({"MAX" if better_low else "MIN"}({rng}),"")'
                ws.cell(r, col, f).number_format = "0.00" if (lower[t] and not is_pct) else "0"
    ws.cell(LAST + 4, 1, "Percentil mais elevado = melhor desempenho relativo à referência da mesma idade. "
            "Nos testes de tempo, menor tempo = melhor; nos de pontos/cestos, maior = melhor.").font = Font(italic=True, size=9)
    # validação, formatação condicional, ajudantes ocultos
    dv = DataValidation(type="whole", operator="between", formula1=str(age_lo), formula2=str(age_hi),
                        showErrorMessage=True, errorTitle="Idade fora do intervalo",
                        error=f"Introduza uma idade entre {age_lo} e {age_hi} anos.")
    ws.add_data_validation(dv); dv.add(f"D{FIRST}:D{LAST}")
    dvn = DataValidation(type="decimal", operator="greaterThan", formula1="0", showErrorMessage=True,
                         errorTitle="Valor inválido", error="Introduza um número positivo.")
    ws.add_data_validation(dvn)
    for label, _, vcol in tests:
        dvn.add(f"{L(vcol)}{FIRST}:{L(vcol)}{LAST}")
        rng = f"{L(vcol + 1)}{FIRST}:{L(vcol + 1)}{LAST}"
        ws.conditional_formatting.add(rng, ColorScaleRule(start_type="num", start_value=0, start_color="F8696B",
                                                          mid_type="num", mid_value=50, mid_color="FFEB84",
                                                          end_type="num", end_value=100, end_color="63BE7B"))
    for h in range(helper0, helper0 + 2 * len(tests)):
        ws.column_dimensions[L(h)].hidden = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "4:4"
    if example:
        for i, row in enumerate(example):
            r = FIRST + i
            for c, v in enumerate(row, 2):
                ws.cell(r, c, v)
    return tests, helper0


# ── Normas ──────────────────────────────────────────────────────────────────
def make_norms(ws, blocks):
    title(ws, "Tabelas de referência (percentis por idade)",
          "Valores lidos da Plataforma de Avaliação Quantitativa ENB (mario1950silva-collab.github.io/testes-grau-1). "
          "A coluna F é a usada nos cálculos: onde o valor publicado está fora de ordem, é corrigido (ver coluna G).", "G")
    header(ws, 4, ["Chave", "Teste", "Idade", "Percentil", "Resultado publicado", "Resultado usado no cálculo",
                   "Verificação / nota"])
    for c, w in zip("ABCDEFG", (28, 26, 8, 10, 16, 18, 70)):
        ws.column_dimensions[c].width = w
    r = 5
    notes = {}
    for name, age, pairs, src in blocks:
        for j, (p, v) in enumerate(pairs):
            ws.cell(r, 1, f'=B{r}&"|"&C{r}')
            ws.cell(r, 2, name); ws.cell(r, 3, age); ws.cell(r, 4, p); ws.cell(r, 5, v)
            ws.cell(r, 6, f"=E{r}" if j == 0 else f"=IF(A{r}<>A{r-1},E{r},MAX(E{r},F{r-1}))")
            ws.cell(r, 7, f'=IF(F{r}<>E{r},"Valor publicado fora de ordem; usado "&TEXT(F{r},"0.00")&" no cálculo.",'
                          f'"{notes.get((name, age, p), "")}")')
            for c in range(1, 8):
                ws.cell(r, c).border = BOX
            r += 1
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:G{r - 1}"
    ws.conditional_formatting.add(f"A5:G{r-1}", FormulaRule(formula=['LEN($G5)>0'],
                                  fill=PatternFill("solid", bgColor="FFE699")))
    ws.cell(r + 1, 1, "Referências: " + "; ".join(sorted({f"{b[0]} — {b[3]}" for b in blocks}))).font = Font(italic=True, size=9)
    ws.cell(r + 2, 1, "P10 («<10») registado como 5, como na ficha original; o percentil máximo (90) vale também para "
                      "resultados melhores que o P90.").font = Font(italic=True, size=9)


# ── Ficha individual ────────────────────────────────────────────────────────
def make_ficha(ws, tests, helper0):
    from openpyxl.utils import get_column_letter as L
    title(ws, "FICHA INDIVIDUAL — AVALIAÇÃO TÉCNICA", "Escolha o N.º do atleta na célula C3. Dados vêm da folha Registo.", "G")
    ws["B3"], ws["C3"] = "N.º do atleta", 1
    ws["B3"].font = Font(bold=True)
    ws["C3"].fill = PatternFill("solid", fgColor=SAND); ws["C3"].border = BOX; ws["C3"].alignment = Alignment(horizontal="center")
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2=str(N_ROWS), showErrorMessage=True)
    ws.add_data_validation(dv); dv.add("C3")
    rr = f"MATCH($C$3,Registo!$A${FIRST}:$A${LAST},0)"
    info = [("Nome", "B"), ("Clube", "C"), ("Idade (anos)", "D")]
    for i, (lab, col) in enumerate(info):
        ws.cell(4 + i, 2, lab).font = Font(bold=True)
        ws.cell(4 + i, 3, f'=IFERROR(IF(INDEX(Registo!${col}${FIRST}:${col}${LAST},{rr})="","",INDEX(Registo!${col}${FIRST}:${col}${LAST},{rr})),"")')
        ws.cell(4 + i, 3).alignment = Alignment(horizontal="left")
    header(ws, 8, ["", "Teste", "Resultado", "Percentil", "Melhor referência (P90)", "Mediana (P50)", "Limite P10 (<10)"])
    ws["A8"].fill = PatternFill(fill_type=None); ws["A8"].border = Border()
    spec = [("Drible (s)", 5, 0, True, helper0), ("Defesa (s)", 7, 1, True, helper0 + 2),
            ("Lançamento (pontos)", 9, 2, False, helper0 + 4)]
    for k, (lab, vcol, _, low, scol) in enumerate(spec):
        r = 9 + k
        reg = lambda col: f"INDEX(Registo!${L(col)}${FIRST}:${L(col)}${LAST},{rr})"
        ws.cell(r, 2, lab).font = Font(bold=True)
        ws.cell(r, 3, f'=IFERROR(IF({reg(vcol)}="","",{reg(vcol)}),"")')
        ws.cell(r, 4, f'=IFERROR(IF({reg(vcol+1)}="","",{reg(vcol+1)}),"")').number_format = "0"
        st = reg(scol)
        def ref(off):
            return f'=IFERROR(IF({st}="","",INDEX(Normas!$F:$F,{st}+{off})),"")'
        ws.cell(r, 5, ref(0 if low else 8)); ws.cell(r, 6, ref(4)); ws.cell(r, 7, ref(8 if low else 0))
        for c in range(2, 8):
            ws.cell(r, c).border = BOX
            ws.cell(r, c).alignment = Alignment(horizontal="center" if c > 2 else "left")
        for c in (3, 5, 6, 7):
            ws.cell(r, c).number_format = "0.00" if low else "0"
    ws["B13"] = ("Leitura: o percentil situa o atleta face a jogadores da mesma idade na tabela de referência "
                 "(mais alto = melhor). Cada teste tem a sua unidade, por isso não há nota global.")
    ws["B13"].font = Font(italic=True, size=9)
    ws.merge_cells("B13:G13"); ws["B13"].alignment = Alignment(wrap_text=True); ws.row_dimensions[13].height = 28
    ch = BarChart(); ch.type = "col"; ch.title = "Perfil por percentil"; ch.legend = None
    ch.add_data(Reference(ws, min_col=4, min_row=8, max_row=11), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=2, min_row=9, max_row=11))
    ch.y_axis.scaling.min, ch.y_axis.scaling.max = 0, 100
    ch.y_axis.title = "Percentil"; ch.y_axis.delete = False; ch.x_axis.delete = False
    ch.dataLabels = DataLabelList(); ch.dataLabels.showVal = True
    ch.dataLabels.showSerName = ch.dataLabels.showCatName = ch.dataLabels.showLegendKey = ch.dataLabels.showPercent = False
    ch.height, ch.width = 7.5, 15
    ws.add_chart(ch, "B15")
    for c, w in zip("ABCDEFG", (3, 24, 14, 12, 22, 16, 20)):
        ws.column_dimensions[c].width = w
    ws.page_setup.orientation = "portrait"; ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def make_instructions(ws):
    title(ws, "INSTRUÇÕES E CRITÉRIOS DE AVALIAÇÃO", "Ficha de Avaliação Técnica — 8 a 16 anos", "B")
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 30, 110
    rows = [
        ("1. Folhas", "Registo (8–16 anos: Drible, Defesa, Lançamento) · Registo Bateria 13-16 (testes IV a VII) · "
                      "Ficha Individual (impressão por atleta) · Normas · Exemplo."),
        ("2. Preenchimento", "No Registo, introduza nome, clube, idade e o melhor resultado de cada teste (até 25 atletas). "
                             "A idade aceita 8 a 16 (Registo) ou 13 a 16 (Bateria). Células cinzentas/brancas são de entrada."),
        ("3. Drible e Defesa", "Tempo em segundos; três tentativas cronometradas com 5 min de recuperação; conta o melhor tempo."),
        ("4. Lançamento 8–10 anos", "Pontos Marcados (1 minuto): tabela a 2,60 m, cinco posições a 2,74 m; registar o total de pontos."),
        ("5. Lançamento 11–16 anos", "One Minute Shooting Test: pontuação ponderada (1, 2 ou 3 pontos); melhor de duas tentativas. "
                                     "A ficha escolhe a tabela certa pela idade."),
        ("6. Percentis", "Calculados por interpolação linear entre os patamares publicados para a idade. Resultado melhor que o "
                         "P90 devolve 90; pior que o limite «<10» devolve 5 (convenção da ficha original). Percentil mais alto = melhor."),
        ("7. Bateria 13-16", "Illinois com Drible, Lançamento 1 Minuto, Velocidade e Coordenação e Lances Livres. O documento ENB "
                             "define esta bateria para o Sub-14 (13–14 anos); as tabelas dos 15 e 16 anos vêm da mesma fonte e "
                             "podem ser usadas por decisão do treinador."),
        ("8. Valores a confirmar", "Na folha Normas, a linha destacada a amarelo: Defesa, 15 anos (P70 publicado 7,54 s, melhor que o "
                                   "P80 de 7,70 s, o que é incoerente; usa-se 7,70 s no cálculo). Confirmar na fonte original."),
        ("9. Limites das normas", "As tabelas são específicas dos testes e das amostras publicadas; não comparar resultados obtidos "
                                  "com protocolos diferentes. Lançamento 1 Minuto, Lances Livres: tabelas de jogadores masculinos. "
                                  "Idade sem tabela mostra «Sem norma». Os 11–16 anos de Lançamento e Lances Livres não têm norma feminina."),
        ("10. Referências", "Matulaitis et al. (2019) — Drible, Defesa, Pontos Marcados; Getchell et al. (1998) / Sabonis Basketball "
                            "Center tab. 3.20 — Illinois; T-Test adaptado (Sabonis); Stonkus (2002) tab. S6 — Lances Livres; "
                            "One Minute Shooting tab. S7. Transcrição: Plataforma de Avaliação Quantitativa ENB."),
    ]
    for i, (a, b) in enumerate(rows, 3):
        ws.cell(i, 1, a).font = Font(bold=True, color=NAVY)
        ws.cell(i, 2, b).alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(i, 1).alignment = Alignment(vertical="top")
        ws.row_dimensions[i].height = 18 * (1 + len(b) // 105)


def main(site_html, out):
    site = load_site_norms(site_html)
    blocks, _ = build_blocks(site)
    wb = Workbook()
    ws_reg = wb.active; ws_reg.title = "Registo"
    tests, helper0 = make_register(ws_reg, "main")
    ws_bat = wb.create_sheet("Registo Bateria 13-16"); make_register(ws_bat, "bat")
    ws_f = wb.create_sheet("Ficha Individual"); make_ficha(ws_f, tests, helper0)
    ws_n = wb.create_sheet("Normas"); make_norms(ws_n, blocks)
    ws_e = wb.create_sheet("Exemplo")
    make_register(ws_e, "main", example=[
        ("Atleta A", "Clube Exemplo", 8, 10.0, None, 10.4, None, 16, None),
        ("Atleta B", "Clube Exemplo", 10, 9.0, None, 10.1, None, 26, None),
        ("Atleta C", "Clube Exemplo", 12, 8.3, None, 9.7, None, 6, None),
        ("Atleta D", "Clube Exemplo", 14, 8.9, None, 8.1, None, 9, None),
        ("Atleta E", "Clube Exemplo", 16, 7.9, None, 7.6, None, 5, None)])
    ws_i = wb.create_sheet("Instruções"); make_instructions(ws_i)
    wb.calculation.fullCalcOnLoad = True
    wb.save(out)
    print("guardado:", out, "| blocos de norma:", len(blocks))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
