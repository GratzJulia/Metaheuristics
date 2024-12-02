from random import randint, sample
from representacao import Grafo
from ReadData import read_DIMACS
import matplotlib.pyplot as plt
import networkx as nx

def printGrafo(v: int, a: int, colors: list = ['white']):
    G = nx.Graph()
    [G.add_node(i+1) for i in range(v)]
    [G.add_edge(arestas[i][0], arestas[i][1]) for i in range(a)]

    nx.draw(G, with_labels=True, node_color=colors, node_size=400, edgecolors='black', linewidths=1)
    plt.show()

class AlgoritmoGenetico:
    def __init__(self, grafo: Grafo):
        self.grafo = grafo

    def funcao_objetivo(self, cromossomo):
        penalidade_aresta = 0
        cor_count = {}

        # 1. Violação de restrições
        for aresta in self.grafo.arestas:
            if cromossomo[aresta.origem - 1] == cromossomo[aresta.destino - 1]:
                penalidade_aresta += 20

        # 2. Contagem das cores usadas
        for i, cor in enumerate(cromossomo):
            if cor not in cor_count:
                cor_count[cor] = 0
            cor_count[cor] += 1

        # 3. Desequilíbrio nas cores
        max_count = max(cor_count.values(), default=0)
        min_count = min(cor_count.values(), default=0)
        desequilibrio = max_count - min_count

        return penalidade_aresta + 0.1 * len(cor_count) + 0.5 * desequilibrio

    def fitness(self, cromossomo):
        return -1 * self.funcao_objetivo(cromossomo)

    def construtivo_aleatorio(self, tam_populacao):
        populacao = []
        for _ in range(tam_populacao):
            cromossomo = [randint(1, self.grafo.V) for _ in range(self.grafo.V)]
            populacao.append(cromossomo)
        return populacao

    def crossover(self, pai1, pai2):
        ponto_corte = randint(1, len(pai1) - 1)
        filho = pai1[:ponto_corte] + pai2[ponto_corte:]
        return filho

    def mutacao(self, cromossomo):
        idx = randint(0, len(cromossomo) - 1)
        cromossomo[idx] = randint(1, len(cromossomo) - 1)

    def selecionar_pais(self, populacao, fitness_populacao):
        # Seleção por torneio
        pais = []
        for _ in range(len(populacao) // 2):
            selecionados = sample(list(zip(populacao, fitness_populacao)), 2)
            pais.append(max(selecionados, key=lambda x: x[1])[0])  # Seleciona o melhor pai

        # Se o número de pais for ímpar, remove o último pai
        if len(pais) % 2 != 0:
            pais = pais[:-1]
        return pais

    def execute(self, geracoes=100, tamanho_populacao=50):
        populacao = self.construtivo_aleatorio(tamanho_populacao)

        for geracao in range(geracoes):
            fitness_populacao = [self.fitness(cromossomo) for cromossomo in populacao]

            melhores_pais = self.selecionar_pais(populacao, fitness_populacao)

            nova_populacao = []
            for i in range(0, len(melhores_pais), 2):
                pai1, pai2 = melhores_pais[i], melhores_pais[i + 1]
                filho1 = self.crossover(pai1, pai2)
                filho2 = self.crossover(pai2, pai1)
                self.mutacao(filho1)
                self.mutacao(filho2)
                nova_populacao.append(filho1)
                nova_populacao.append(filho2)

            if len(nova_populacao) > 0:
                populacao = nova_populacao

        melhor_individuo = min(populacao, key=lambda cromossomo: self.funcao_objetivo(cromossomo))
        return melhor_individuo


if __name__ == "__main__":
    v, a, arestas = read_DIMACS('./input-data/S10.txt')
    printGrafo(v, a)

    g = Grafo(v)
    [g.add_aresta(arestas[i][0], arestas[i][1], 0.0) for i in range(a)]
    
    ag = AlgoritmoGenetico(g)
    melhor_colocacao = ag.execute(geracoes=10, tamanho_populacao=6)
    print(melhor_colocacao)
    printGrafo(v, a, melhor_colocacao)
