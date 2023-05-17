Como organizar os movimentos:
- Fazer shuffle pela comida, iterar e verificar se tem agentes à volta até encontrar 2 que não tenham comido ainda
- Ordenar aleatoriamente os agentes (função shuffle)
- Para cada agente, procurar (no basic_agent.py) as comidas mais próximas (em vez de só uma)
    - Com isso, obter uma queue de direções possíveis
- No step, **se não tiver comido**, iterar pela lista de direções possíveis até ser encontrado um movimento possível
    - Comida também conta como ocupada!