from random import choices, randint, random, sample
from collections import Counter
from representacao import Grafo, Individuo
from ReadData import read_DIMACS
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def printGrafo(v: int, a: int, title: str, colors: list = ['white']):
    G = nx.Graph()
    [G.add_node(i+1) for i in range(v)]
    [G.add_edge(arestas[i][0], arestas[i][1]) for i in range(a)]

    plt.title(title)
    nx.draw(G, with_labels=True, node_color=colors, node_size=450, edgecolors='black', linewidths=1)
    plt.show()

class AlgoritmoGenetico:
    def __init__(self, grafo: Grafo):
        self.grafo = grafo
        self.tamanho_populacao = 2 * grafo.V
        self.len_elites = 0.2 * self.tamanho_populacao
        self.len_nao_elites = self.tamanho_populacao - self.len_elites

    def FO(self, cromossomo) -> float:
        penalidade_aresta = 0
        for a in self.grafo.arestas:
            if cromossomo[a.origem - 1] == cromossomo[a.destino - 1]:
                penalidade_aresta += 1

        qtd_cores = len(set(cromossomo))

        cor_count = Counter(cromossomo)
        floor_val = self.grafo.V // qtd_cores
        ceil_val = floor_val + (1 if self.grafo.V % qtd_cores != 0 else 0)

        distribution_penalty = 0
        for ccount in cor_count.values():
            if ccount < floor_val:
                distribution_penalty += (floor_val - ccount) ** 2
            elif ccount > ceil_val:
                distribution_penalty += (ccount - ceil_val) ** 2

        distribution_penalty *= 100
        return float(penalidade_aresta * 1000 + qtd_cores * 100 + distribution_penalty)

    def fitness(self, cromossomo):
        return -1 * self.FO(cromossomo)

    def construtivo_aleatorio(self):
        populacao = []
        for _ in range(self.tamanho_populacao):
            cromossomo = [randint(1, self.grafo.V) for _ in range(self.grafo.V)]
            populacao.append(Individuo(cromossomo, self.fitness(cromossomo)))
        return populacao

    def set_elite(self, populacao: list):
        fitness_ordenado = sorted(populacao, key=lambda i: i.value, reverse=True)
        melhores = fitness_ordenado[:int(self.len_elites)]
        piores = fitness_ordenado[int(self.len_elites):]
        return melhores, piores

    def crossover(self, pai1, pai2):
        ponto_corte = randint(1, len(pai1.cromossomo) - 1)
        filho = pai1.cromossomo[:ponto_corte] + pai2.cromossomo[ponto_corte:]
        return Individuo(filho, self.fitness(filho))

    def mutacao(self, individuo: Individuo):
        gene_aleatorio = randint(0, len(individuo.cromossomo) - 1)
        individuo.cromossomo[gene_aleatorio] = randint(1, len(individuo.cromossomo) - 1)
        individuo.value = self.fitness(individuo.cromossomo)

    def torneio_numpy(self, populacao, N, q = 2):
        fitness = np.array([ind.value for ind in populacao])
        torneios = np.random.choice(len(populacao), size=(N, q), replace=True)
        vencedores_indices = torneios[np.arange(N), np.argmax(fitness[torneios], axis=1)]
        return [populacao[idx] for idx in vencedores_indices]

    def roleta(self, populacao, fitness_populacao):
        min_fitness = min(fitness_populacao)
        ajustado = [f - min_fitness + 1 for f in fitness_populacao]
        pais = choices(populacao, weights=ajustado, k=len(populacao) // 2)
        return pais

    def execute(self, geracoes=100):
        populacao = self.construtivo_aleatorio()
        
        for geracao in range(geracoes):
            elites, nao_elites = self.set_elite(populacao)
            nova_populacao = elites.copy()
            novos_individuos = []

            while len(novos_individuos) < self.len_nao_elites:
                probabilidade = random()
                if probabilidade < 0.94:
                    pais = self.torneio_numpy(populacao, 2)
                    filho1 = self.crossover(pais[0], pais[1])
                    filho2 = self.crossover(pais[1], pais[0])
                    novos_individuos.append(filho1)
                    novos_individuos.append(filho2)
                else:
                    aleatorio = sample(nao_elites, 1)
                    self.mutacao(aleatorio[0])

            uniao = populacao + novos_individuos
            escolhidos = self.torneio_numpy(uniao, len(nao_elites))
            selecionados = nova_populacao + escolhidos
            populacao.clear()
            populacao.extend(selecionados)

        melhor_individuo = min(populacao, key=lambda ind: self.FO(ind.cromossomo))
        return melhor_individuo


if __name__ == "__main__":
    v, a, arestas = read_DIMACS('./input-data/S10.txt')

    g = Grafo(v)
    [g.add_aresta(arestas[i][0], arestas[i][1], 0.0) for i in range(a)]
    
    printGrafo(v, a, str(g))

    ag = AlgoritmoGenetico(g)
    melhor_colocacao: Individuo = ag.execute(geracoes=40)
    print(melhor_colocacao.cromossomo)
    print(melhor_colocacao.value)
    printGrafo(v, a, str(g), melhor_colocacao.cromossomo)
