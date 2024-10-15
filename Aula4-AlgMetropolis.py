
class Grafo:
    def __init__(self, vertices):
        self.V = vertices
        self.arestas = []

    def add_aresta(self, i, j, custo):
        self.arestas.append((custo, i, j))

def kruskal(G: Grafo):
    G.arestas.sort()
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

    for peso, i, j in G.arestas:
        forma_ciclo = find(i) == find(j)
        if not forma_ciclo:
            union(i, j)
            AGM.append((i, j, peso))
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
    [print(f"{i}-{j}: {custo}") for i, j, custo in agm]
