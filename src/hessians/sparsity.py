"""Pattern di sparsità (CPR / coloring) per le Hessiane dei problemi di test.

Riferimento: slide del corso NO4LSP sulla "GENERALIZATION OF THE SCHEME:
COLORING GRAPHS". Costruito il grafo
    V = {x_1, ..., x_n},
    {x_i, x_j} in E  <=>  esiste k tale che g_k dipende sia da x_i sia da x_j,
le variabili dello stesso colore hanno supporti delle colonne H[:,j] disgiunti
e possono essere perturbate insieme da un'unica chiamata al gradiente.

Per i due problemi del nostro studio i pattern si ottengono per ispezione,
quindi non serve un risolutore generale di coloring (cfr. PLAN.md sezione
"Out of scope").
"""


def cpr_groups_p16(n):
    """Problema 16 (Banded Trigonometric): Hessiana DIAGONALE.

    Ogni g_j dipende solo da x_j, dunque il grafo G non ha archi: un solo
    colore basta. Tutte le variabili in un unico gruppo => UNA sola coppia
    di chiamate al gradiente per estrarre l'intera diagonale (cfr.
    hess_fd_diag), indipendentemente da n.
    """
    return [list(range(n))]


def cpr_groups_p28(n):
    """Problema 28 (Variably Dimensioned): Hessiana PIENA.

    Ogni g_j dipende da TUTTE le x_i: il grafo di sparsita' e'
    completo, servono n colori distinti e la coloring non produce
    risparmio. Esposto qui per simmetria con P16 e per documentare
    esplicitamente l'assenza di guadagno strutturale.

    Riportato nel report: per questo problema le uniche ottimizzazioni
    della FD-Hessiana sono quelle "general purpose":
        - simmetrizzazione del risultato   (H_fd <- (H_fd + H_fd^T)/2)
        - uso del gradiente analitico, O(n) per chiamata, anziche'
          una FD sul valore di f, che costerebbe O(n^2)
        - path matrix-free Hv per il Truncated Newton (un solo
          gradiente per ogni prodotto Hv, senza materializzare la
          matrice n x n).
    """
    return [[j] for j in range(n)]
