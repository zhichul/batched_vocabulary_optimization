import math


SEP = "-"
class Cell:
    pass

def viterbi_edit_distance_segmentation(surf, deep):
    """
    minimize edit distance
    (M \ N) b i o l o g y
    b
    i
    o
    -
    o
    l
    o
    g
    y
    """
    M = len(deep)
    N = len(surf)
    table = [[Cell() for _ in range(N)] for _ in range(M)]
    table[0][0].alpha = 0
    table[0][0].back_pointers = None
    for i in range(M):
        for j in range(N):
            if i == j == 0:
                continue
            #option 1
            a1 = math.inf
            bp1 = (i-1, j) #del
            a2 = math.inf
            bp2 = (i, j-1) #insert
            a3 = math.inf
            bp3 = (i-1, j-1) #replace or match
            if deep[i] == SEP:
                # only can delete seperator
                table[i][j].alpha = table[bp1[0]][bp1[1]].alpha # no cost
                table[i][j].back_pointers = [bp1] # no cost
            else:
                if bp1[0] >= 0:
                    a1 = table[bp1[0]][bp1[1]].alpha + 1
                if bp2[1] >= 0:
                    a2 = table[bp2[0]][bp2[1]].alpha + 1
                if bp3[0] >= 0 and bp3[1] >= 0:
                    a3 = table[bp3[0]][bp3[1]].alpha + (0 if deep[i] == surf[j] else 1)
                table[i][j].alpha = min(a1, a2, a3)
                table[i][j].back_pointers = []
                for a, bp in zip([a1, a2, a3], [bp1, bp2, bp3]):
                    if a == table[i][j].alpha:
                        table[i][j].back_pointers.append(bp)
    # for i in range(M):
    #     for j in range(N):
    #         print(i,j,table[i][j].back_pointers, table[i][j].alpha)
    # build all the viterbi segmentations
    for fences in set(dfs(deep, surf, table, M-1, N-1)):
        print(split(surf, fences))

def dfs(deep, surf, table, i, j):
    if i == j == 0:
        yield tuple()
    else:
        if deep[i] == SEP:
            for prefix in dfs(deep, surf, table, *table[i][j].back_pointers[0]):
                yield  prefix + (j,)
        else:
            for bp in table[i][j].back_pointers:
                for prefix in dfs(deep, surf, table, *bp):
                    yield prefix

def split(surf, fences):
    fences = set(fences)
    out = []
    for i, c in enumerate(surf):
        out.append(c)
        if i in fences:
            out.append(SEP)
    return "".join(out)

if __name__ == "__main__":
    viterbi_edit_distance_segmentation("bacteriologists", "bacterium-ology-ist-s")
    viterbi_edit_distance_segmentation("biologists", "bio-ology-ist-s")