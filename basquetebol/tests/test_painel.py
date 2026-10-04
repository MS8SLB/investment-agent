def test_painel(client):
    r = client.get("/")
    assert r.status_code == 200 and "Atletas avaliados" in r.text and "Indicadores por teste" in r.text
