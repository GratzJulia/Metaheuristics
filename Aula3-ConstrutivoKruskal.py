class Aresta:
    def __init__(self, custo, i, j):
        self.origem: int = i
        self.destino:int = j
        self.custo: float = custo

class Grafo:
    def __init__(self, vertices):
        self.V: int = vertices
        self.arestas = []

    def add_aresta(self, i, j, custo):
        self.arestas.append(Aresta(custo, i, j))

def kruskal(G: Grafo):
    # Heuristica Construtiva
    G.arestas.sort(key=lambda a: a.custo)
    pai = {}
    rank = {}

    for j in range(1, G.V+1):
        pai[j] = j
        rank[j] = 0

    def find(v):
        if pai[v] != v:
            pai[v] = find(pai[v])
        return pai[v]

    def union(i, j):
        raiz_i = find(i)
        raiz_j = find(j)

        if raiz_i != raiz_j:
            # União por rank
            if rank[raiz_i] > rank[raiz_j]:
                pai[raiz_j] = raiz_i
            elif rank[raiz_i] < rank[raiz_j]:
                pai[raiz_i] = raiz_j
            else:
                pai[raiz_j] = raiz_i
                rank[raiz_i] += 1

    AGM = []  # Arvore Geradora Minima
    count_aresta = 0

    for aresta in G.arestas:
        forma_ciclo = find(aresta.origem) == find(aresta.destino)
        if not forma_ciclo:
            union(aresta.origem, aresta.destino)
            AGM.append(aresta)
            count_aresta += 1

        if count_aresta == G.V - 1: break

    return AGM

if __name__ == "__main__":
    g = Grafo(7)    # grafo de exemplo do slide
    g.add_aresta(1, 3, 6)
    g.add_aresta(2, 3, 3)
    g.add_aresta(2, 5, 8)
    g.add_aresta(2, 6, 3)
    g.add_aresta(3, 4, 9)
    g.add_aresta(3, 6, 1)
    g.add_aresta(4, 7, 1)
    g.add_aresta(5, 6, 2)
    g.add_aresta(6, 7, 5)

    agm = kruskal(g)
    print("Arestas da Árvore Geradora Mínima:")
    [print(f"{a.origem}-{a.destino}: {a.custo}") for a in agm]
