from random import choices, randint, random, sample
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ReadData import read_DIMACS, instancias
from representacao import Grafo, Individuo

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

    def FO(self, cromossomo):
        penalidade_aresta = 0
        for a in self.grafo.arestas:
            if cromossomo[a.origem - 1] == cromossomo[a.destino - 1]:
                penalidade_aresta += 1

        qtd_cores = len(set(cromossomo))

        cor_count = Counter(cromossomo)
        floor_val = self.grafo.V // qtd_cores
        ceil_val = floor_val + (1 if self.grafo.V % qtd_cores != 0 else 0)

        desequilibrio = 0
        for ccount in cor_count.values():
            if ccount < floor_val:
                desequilibrio += (floor_val - ccount) ** 2
            elif ccount > ceil_val:
                desequilibrio += (ccount - ceil_val) ** 2

        return {"fo": float(penalidade_aresta * 1000 + qtd_cores * 100 + desequilibrio * 10), "p": penalidade_aresta, "c": qtd_cores, "d": desequilibrio}

    def fitness(self, cromossomo):
        obj = self.FO(cromossomo)
        obj['fo'] *= -1
        return obj

    def construtivo_aleatorio(self):
        populacao = []
        for _ in range(self.tamanho_populacao):
            cromossomo = [int(random()*10) for _ in range(self.grafo.V)]
            f = self.fitness(cromossomo)
            populacao.append(Individuo(cromossomo, f['fo'], f['c'], f['p'], f['d']))
        return populacao

    def set_elite(self, populacao: list):
        fitness_ordenado = sorted(populacao, key=lambda i: i.value, reverse=True)
        melhores = fitness_ordenado[:int(self.len_elites)]
        piores = fitness_ordenado[int(self.len_elites):]
        return melhores, piores

    def crossover(self, pai1, pai2):
        ponto_corte = randint(1, len(pai1.cromossomo) - 1)
        filho = pai1.cromossomo[:ponto_corte] + pai2.cromossomo[ponto_corte:]
        f = self.fitness(filho)
        return Individuo(filho, f['fo'], f['c'], f['p'], f['d'])

    def mutacao(self, individuo: Individuo):
        gene_aleatorio = randint(0, len(individuo.cromossomo) - 1)
        individuo.cromossomo[gene_aleatorio] = randint(1, len(individuo.cromossomo) - 1)
        f = self.fitness(individuo.cromossomo)
        individuo.value = f['fo']
        individuo.qtd_cores = f['c']
        individuo.rest_aresta = f['p']
        individuo.desequilibrio = f['d']

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

    def execute(self, geracoes=1000):
        populacao = self.construtivo_aleatorio()
        melhor_inicial = max(populacao, key=lambda ind: ind.value)
        # print('Melhor cromossomo do construtivo:', melhor_inicial.cromossomo)
        print('Qtd cores:', melhor_inicial.qtd_cores)
        print('Penalidade restrição de arestas:', melhor_inicial.rest_aresta)
        print('Desequilibrio de cores:', melhor_inicial.desequilibrio)

        for geracao in range(geracoes):
            elites, nao_elites = self.set_elite(populacao)
            nova_populacao = elites.copy()
            novos_individuos = []

            while len(novos_individuos) < self.len_nao_elites:
                probabilidade_cruz = random()
                probabilidade_mut = random()
                if probabilidade_cruz < 0.94:
                    pais = self.torneio_numpy(populacao, 2)
                    filho1 = self.crossover(pais[0], pais[1])
                    filho2 = self.crossover(pais[1], pais[0])
                    novos_individuos.append(filho1)
                    novos_individuos.append(filho2)
                
                if probabilidade_mut <= 0.05:
                    aleatorio = sample(nao_elites, 1)
                    self.mutacao(aleatorio[0])

            uniao = populacao + novos_individuos
            escolhidos = self.torneio_numpy(uniao, len(nao_elites))
            populacao.clear()
            populacao.extend(nova_populacao + escolhidos)

        melhor_individuo = max(populacao, key=lambda ind: ind.value)
        return melhor_individuo

def main():
    g = Grafo(v)
    [g.add_aresta(arestas[i][0], arestas[i][1], 0.0) for i in range(a)]
    print(g)
    # printGrafo(v, a, str(g))

    ag = AlgoritmoGenetico(g)
    ag_inicio = datetime.now()
    melhor_colocacao: Individuo = ag.execute(geracoes=700)
    ag_fim = datetime.now()
    # print('Melhor cromossomo final: ', melhor_colocacao.cromossomo)
    print('Qtd cores:', melhor_colocacao.qtd_cores)
    print('Penalidade restrição de arestas:', melhor_colocacao.rest_aresta)
    print('Desequilibrio de cores:', melhor_colocacao.desequilibrio)
    print('Tempo de execução: ', ag_fim - ag_inicio)
    print()
    # printGrafo(v, a, str(g) + " \nFitness da solução: " + str(melhor_colocacao.value), melhor_colocacao.cromossomo)

if __name__ == "__main__":
    for nome in instancias():
        v, a, arestas = read_DIMACS('./input-data/' + nome)
        print('Instância: ', nome)
        main()