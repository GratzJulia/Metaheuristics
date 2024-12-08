class Aresta:
    def __init__(self, custo, i, j):
        self.origem: int = i
        self.destino:int = j
        self.custo: float = custo

    def __str__(self):
        return f'Aresta {self.origem}-{self.destino} com custo {self.custo}'


class Vertice:
    def __init__(self):
        self.cor: int = 0
        self.vizinhos = []

    def __str__(self):
        return f'Vertice com cor {self.cor} e {len(self.vizinhos)} {"vizinho" if len(self.vizinhos) == 1 else "vizinhos"}'


class Grafo:
    def __init__(self, vertices: int):
        self.V: int = vertices
        self.arestas = []
        self.vertices = [Vertice() for _ in range(vertices)]

    def __str__(self):
        return f'Grafo com {self.V} vértices e {len(self.arestas)} arestas'

    def add_aresta(self, i, j, custo):
        self.arestas.append(Aresta(custo, i, j))
        if i not in self.vertices[j-1].vizinhos: self.vertices[j-1].vizinhos.append(i)
        if j not in self.vertices[i-1].vizinhos: self.vertices[i-1].vizinhos.append(j)

class Individuo:
    def __init__(self, c, fitness):
        self.value: float = fitness
        self.cromossomo: list = c