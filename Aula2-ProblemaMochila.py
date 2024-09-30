from pandas import read_csv

def canAllItemsBeTaken(Ws: list, capacity: int):
    sum = 0
    for w in Ws:
        sum += w
        if sum > capacity: return False
    
    return True

def solveMochila(len: int, P, W, C: int, s = []):
    sortingCriteria = [{'criteria': float(P[i]/W[i]), 'p': P[i], 'w': W[i]} for i in range(len)]
    sortingCriteria.sort(key=lambda x: x['criteria'], reverse=True)

    for i in range(len):
        if C - sortingCriteria[i]['w'] > 0:
            s.append(i+1)
            C = C - sortingCriteria[i]['w']
    
    return s

def main():
    csv = read_csv('./input-data/TestExample-ProblemaDaMochilaClassico.csv', sep=';')

    lenItems = csv.get('Items').size
    allW = csv.get('w') # grandeza relacionada ao espaco fisico
    allP = csv.get('p') # valor do item (recompensa)
    s = []
    c = (7)

    if canAllItemsBeTaken(allW, c): 
        s = [i+1 for i in range(lenItems)]
        return s
    
    s = solveMochila(lenItems, allP, allW, c)
    print('Solution set: ', s)
    

if __name__ == "__main__":
    main()