def read_DIMACS(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    qtd_v = 0
    qtd_a = 0
    arestas = []

    for line in lines:
        line = line.strip()
        if line.startswith('p'):
            qtd = line.split()
            qtd_v = int(qtd[2])
            qtd_a = int(qtd[3])
            continue
        elif line.startswith('c'):
            continue
        
        data = line.split()
        data.reverse()  # ordem não importa para grafos não direcionados!
        data.pop()
        a = (int(data[0]), int(data[1]))
        arestas.append(a)

    return qtd_v, qtd_a, arestas
