"""Dados iniciais: escalões e as 7 baterias do documento ENB (protocolos transcritos, não alterados).

Não são inseridos valores de referência nem percentis.
"""
from sqlalchemy import select

from .models import Category, CategoryTest, TestDef

CATEGORIES = [("Sub-8", 8, False), ("Sub-10", 10, True), ("Sub-12", 12, True), ("Sub-14", 14, True)]

FIG = "A figura do percurso encontra-se no documento ENB_Avaliacao_Quantitativa."

TESTS = [
    dict(
        code="movimentos_defensivos", numeral="I", name="Teste de Movimentos Defensivos",
        source="Johnson e Nelson, 1986",
        capacity="Deslocamento defensivo — velocidade e técnica de defesa",
        objective="Avaliar a capacidade dos jogadores na execução de movimentos defensivos básicos.",
        description=(
            "O teste é realizado dentro dos limites da linha de lance livre, atrás do cesto, e das linhas da "
            "área de ressalto. Os marcadores centrais da área de ressalto servem como alvos C e F (além disso, "
            "devem ser marcados, com fita adesiva, quatro pontos nos cantos do retângulo delimitado: A, B, D e E).\n"
            "Trajetória: A-B-C-D-E-F-A.\n"
            "O atleta inicia o teste no ponto A, de costas para o cesto. Ao sinal de partida (\"Ready, go\"), "
            "desloca-se lateralmente para a esquerda, sem cruzar os pés, até ao ponto B, onde toca o chão fora "
            "da área com a mão esquerda. Em seguida, executa um drop step e desliza até ao ponto C, tocando o "
            "chão fora da área com a mão direita.\n"
            "O percurso continua conforme indicado na Figura até que ambos os pés cruzem a linha de chegada.\n" + FIG),
        procedure=("Cada atleta realiza três tentativas cronometradas.\n"
                   "O tempo de recuperação entre as tentativas é de 5 minutos.\n"
                   "O melhor tempo registado é considerado para análise."),
        materials="Fita adesiva (marcação dos pontos A, B, D e E); cronómetro; campo com área de lance livre/ressalto.",
        result_note="Tempo (s). Considera-se o melhor tempo das tentativas.",
        unit="s", direction="lower", n_attempts=3, attempts_rule="3 tentativas cronometradas; recuperação de 5 min.",
        decimals=2),
    dict(
        code="drible", numeral="II", name="Teste de Drible", source="Matulaitis, K. et al., 2019",
        capacity="Manuseamento de bola em movimento, mão não dominante",
        objective="Medir as habilidades de manuseamento da bola em movimento.",
        description=(
            "Seis cones são dispostos na área do lance livre de um campo de basquetebol para servir como "
            "obstáculos (Figura 1A).\n"
            "Ao sinal \"Pronto, vai\", o executante começa a driblar com a mão não dominante do lado não "
            "dominante do ponto A até ao lado não dominante do ponto B (drible com a mão esquerda).\n" + FIG),
        procedure=("Três tentativas cronometradas são realizadas.\n"
                   "O tempo de recuperação entre as tentativas foi de 5 minutos.\n"
                   "O melhor resultado foi utilizado para análise."),
        materials="6 cones; bola de basquetebol; cronómetro; área de lance livre.",
        result_note="Tempo (s). Considera-se o melhor resultado.",
        unit="s", direction="lower", n_attempts=3, attempts_rule="3 tentativas cronometradas; recuperação de 5 min.",
        decimals=2),
    dict(
        code="pontos_marcados", numeral="III", name="Teste de Pontos Marcados",
        source="Matulaitis, K. et al., 2019",
        capacity="Precisão de finalização em contexto de curta duração",
        objective="Avaliar a capacidade de finalização e precisão de lançamento em contexto de curta duração (1 minuto).",
        description=(
            "Lançamento em tabela de minibasquete a 2,60 a partir de 5 posições definidas a uma distância de "
            "2,74 metros. Cada praticante tem 1 minuto. Só passa para a posição seguinte quando marcar na anterior."),
        procedure=("O documento ENB não indica o número de tentativas nem a forma de contabilizar o resultado.\n"
                   "Valores assumidos nesta aplicação (A CONFIRMAR pelo responsável metodológico): "
                   "1 tentativa de 1 minuto; resultado = número de cestos convertidos (maior = melhor)."),
        materials="Cesto de minibasquete (2,60 m), bola, cronómetro, marcações das 5 posições a 2,74 m.",
        result_note="Pontos/cestos convertidos em 1 minuto (maior = melhor). Unidade a confirmar.",
        unit="pontos", direction="higher", n_attempts=1,
        attempts_rule="Não especificado no documento ENB — a confirmar.", attempts_confirmed=False, decimals=0),
    dict(
        code="illinois_drible", numeral="IV", name="Teste de Illinois com Drible", source="Getchell et al., 1998",
        capacity="Agilidade, mudanças de direção e controlo de bola em drible",
        objective=("Avaliar a agilidade do jogador enquanto dribla a bola, combinando velocidade de deslocamento, "
                   "coordenação, mudanças rápidas de direção e controlo da bola."),
        description=(
            "O percurso tem aproximadamente 10 metros de comprimento entre a zona inicial e a zona final, com os "
            "pontos de mudança de direção identificados pelas letras A, B, C, D e E. O jogador inicia em A.\n"
            "A sequência do percurso é: A → B → C → D → E → D → C → B → F (chegada). O jogador não percorre "
            "simplesmente os 10 metros em linha reta — tem de realizar sucessivas mudanças de direção, mantendo "
            "sempre o drible.\n"
            "O atleta realiza o percurso Illinois enquanto dribla a bola, procurando completar todo o circuito no "
            "menor tempo possível. O tempo é iniciado e terminado através de sinal sonoro do treinador.\n" + FIG),
        procedure=("Cada participante realiza uma tentativa. Se o jogador perder o controlo da bola, o teste é "
                   "repetido até um máximo de três tentativas. Para a análise, é utilizado o melhor resultado obtido."),
        materials="Campo de basquetebol; percurso do Illinois Agility Test; sistema de cronometragem; bola de basquetebol.",
        result_note="Tempo total, em segundos (s), para completar o percurso. Quanto menor o tempo, melhor o desempenho.",
        unit="s", direction="lower", n_attempts=3,
        attempts_rule="1 tentativa; repetir (máx. 3) apenas se o jogador perder o controlo da bola.", decimals=2),
    dict(
        code="one_minute_shooting", numeral="V", name="One Minute Shooting Test (Teste de Lançamento de 1 Minuto)",
        source="Documento ENB",
        capacity="Precisão de lançamento a partir de diferentes zonas, sob fadiga",
        objective=("Avaliar a precisão dos lançamentos ao cesto a partir de diferentes zonas do campo, sob "
                   "condições de elevada intensidade física."),
        description=(
            "Preparação: numa das metades do campo são colocados dois cones: um junto à linha de três pontos, "
            "num ângulo de 45° em relação ao plano da tabela, e outro junto à extremidade da linha de lance "
            "livre, do lado oposto. Um ajudante, colocado debaixo do cesto, passa a bola ao praticante.\n"
            "Ao sinal, o praticante corre desde o meio-campo em direção ao cesto e, depois de receber a bola, "
            "realiza um lançamento em dois tempos.\n"
            "Após o lançamento, corre de volta até ao círculo central, toca na linha do círculo e volta a "
            "acelerar em direção ao cone colocado junto ao lado exterior da linha de lance livre.\n"
            "Depois de receber a bola, realiza um lançamento em suspensão e corre novamente até ao círculo "
            "central, tocando na sua linha.\n"
            "De seguida, acelera novamente em direção ao cone colocado junto à linha de três pontos. Depois de "
            "receber a bola, realiza um lançamento ao cesto e regressa ao círculo central.\n"
            "Esta sequência de lançamentos é repetida durante 1 minuto."),
        procedure=("O teste é realizado duas vezes, trocando o lado de realização, sendo considerado o melhor "
                   "resultado.\nPontuação: 1 ponto — lançamento convertido junto ao cesto; 2 pontos — convertido a "
                   "partir da zona do cone junto à linha de lance livre; 3 pontos — convertido a partir da linha "
                   "de três pontos. Os pontos são contabilizados pelo treinador ou jogador que realiza os passes."),
        materials="Dois cones; uma bola; cronómetro; um ajudante para passar a bola.",
        result_note="Soma total dos pontos obtidos em 1 minuto (maior = melhor).",
        unit="pontos", direction="higher", n_attempts=2, attempts_rule="2 realizações, trocando o lado.", decimals=0),
    dict(
        code="velocidade_coordenacao", numeral="VI", name="Teste de Velocidade e Coordenação (Defensive Movement Test)",
        source="Documento ENB",
        capacity="Velocidade e técnica de deslizamento defensivo",
        objective="Avaliar as capacidades físicas de velocidade e coordenação dos jogadores de basquetebol.",
        description=(
            "Ao sinal, o participante:\n"
            "• corre para a frente, a partir da linha final, o mais rapidamente possível até ao cone central;\n"
            "• desloca-se lateralmente, em passo de deslocamento para a direita, durante 5 m, até ao cone da direita;\n"
            "• desloca-se lateralmente, em passo de deslocamento para a esquerda, durante 10 m, até ao cone da esquerda;\n"
            "• regressa lateralmente para a direita até ao cone central;\n"
            "• finalmente, desloca-se de costas até alcançar a linha de chegada."),
        procedure="Regista-se o melhor tempo obtido em duas tentativas.",
        materials="5 cones e um cronómetro.",
        result_note=("Tempo (s). NOTA: o documento inclui uma tabela de percentis do T-Test (12–20 anos); não "
                     "corresponde a este protocolo e NÃO é aplicada como referência."),
        unit="s", direction="lower", n_attempts=2, attempts_rule="2 tentativas.", decimals=2),
    dict(
        code="lancamentos_livres", numeral="VII", name="Teste de Lançamentos Livres", source="Stonkus, 2002",
        capacity="Precisão e estabilidade do lançamento livre",
        objective="Avaliar a precisão e a estabilidade da capacidade de lançamento de lances livres.",
        description=(
            "O atleta executa um lance livre; no primeiro e no segundo lançamento, a bola é passada por um "
            "companheiro, após o terceiro lançamento o próprio atleta recupera a bola, dribla até à linha de "
            "lance livre e volta a lançar. Este processo é repetido até serem realizados 30 lances livres.\n"
            "O atleta deve efetuar o lançamento até 5 segundos após o momento em que o companheiro lhe passa a "
            "bola ou após recuperar a bola e se posicionar na linha de lance livre."),
        procedure="O teste é realizado uma única vez.",
        materials="Cesto, bola, um companheiro para passar a bola, contagem de lançamentos.",
        result_note="Número de lançamentos convertidos (em 30).",
        unit="cestos (em 30)", direction="higher", n_attempts=1, attempts_rule="1 realização de 30 lançamentos.",
        decimals=0, max_value=30),
]


def seed(db):
    cats = {c.name: c for c in db.scalars(select(Category))}
    for name, sort, enb in CATEGORIES:
        if name not in cats:
            cats[name] = Category(name=name, sort=sort, has_enb_protocol=enb)
            db.add(cats[name])
    db.flush()
    if db.scalar(select(TestDef.id).limit(1)) is None:
        tests = []
        for i, t in enumerate(TESTS, 1):
            td = TestDef(sort=i, is_enb=True, **t)
            db.add(td)
            tests.append(td)
        db.flush()
        # O documento ENB refere Sub10/Sub12/Sub14. Sub-8: sem protocolo -> nenhum teste pré-selecionado.
        for cname in ("Sub-10", "Sub-12", "Sub-14"):
            for td in tests:
                db.add(CategoryTest(category_id=cats[cname].id, test_id=td.id))
    db.commit()
