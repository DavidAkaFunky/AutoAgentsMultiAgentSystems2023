Como organizar os movimentos:
- Fazer shuffle pela comida, iterar e verificar se tem agentes à volta até encontrar 2 que não tenham comido ainda
- Ordenar aleatoriamente os agentes (função shuffle)
- Para cada agente, procurar (no regular_agent.py) as comidas mais próximas (em vez de só uma)
    - Com isso, obter uma queue de direções possíveis
- No step, **se não tiver comido**, iterar pela lista de direções possíveis até ser encontrado um movimento possível
    - Comida também conta como ocupada!

Métricas:
- Natalidade
    - Geral
    - Dividida por tipos (greedy e pacífico)
- Mortalidade
    - Geral
    - Dividida por tipos (greedy e pacífico)
- Energia média
    - Geral
    - Dividida por tipos (greedy e pacífico)

QUANDO O RATIONAL NÃO SABE PARA ONDE IR:
- Fixar um ponto