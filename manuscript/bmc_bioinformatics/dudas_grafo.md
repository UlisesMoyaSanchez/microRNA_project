# Dudas y aclaraciones — estructura del grafo y sesgo de grado

Notas de trabajo para preparar la sección Methods ("Data and graph construction" /
"Evaluation protocol") de `manuscript.tex`. No es un documento canónico de resultados
(ese rol lo cumple `results/EVALUATION_AUDIT.md`); esto es material pedagógico previo a
convertir las viñetas STAGE en prosa.

---

## 1. ¿Qué es un "nodo" en este proyecto?

No hay un único tipo de nodo — el grafo es heterogéneo y tiene **tres tipos simultáneos**
(`data/03_build_graph/build_heterograph.py`):

| Tipo de nodo | Qué representa | Feature que carga |
|---|---|---|
| `miRNA` | un microRNA | embedding aprendido (inicializado al azar) |
| `gene` | un gen (los más variables del scRNA-seq) | expresión media log-normalizada (1 número) |
| `cell` | una célula individual | PCA de su expresión (50 dimensiones) |

Cuál es "el nodo relevante" depende de la tarea:

- **Link prediction** (resultado central del paper): protagonistas `miRNA` y `gene`. La
  arista predicha es `(miRNA, regulates, gene)`.
- **Clasificación de tipo celular** (tarea de control): protagonista `cell`.
- `gene` es infraestructura compartida entre ambas tareas.

## 2. ¿Qué es una "arista"?

Tres tipos (cada una con su reversa automática para el paso de mensajes):

1. `(miRNA, regulates, gene)` — de miRDB/miRTarBase. La que el modelo predice.
2. `(cell, expresses, gene)` — expresión log-normalizada por encima de un umbral (dispersa).
3. `(gene, coexpressed_with, gene)` — correlación de Pearson por encima de un umbral.

## 3. Ejemplo de juguete: nodo, arista, grado, matriz de adyacencia

```
miRNAs:  m1, m2, m3
genes:   g1, g2, g3, g4

Aristas (m regula g):
m1 → g1
m1 → g2
m2 → g1
m3 → g1
m3 → g4
```

**Grado** = cuántas aristas toca cada nodo:
- `g1` grado **3** (lo regulan m1, m2, m3) — gen "popular"
- `g2` grado **1**, `g4` grado **1**, `g3` grado **0**
- `m1` grado 2, `m2` grado 1, `m3` grado 2

**Matriz de adyacencia** (miRNA × gene, bipartita):

```
        g1   g2   g3   g4
   m1 [  1    1    0    0  ]
   m2 [  1    0    0    0  ]
   m3 [  1    0    0    1  ]
```

Suma de columna = grado del gen; suma de fila = grado del miRNA.

Ventaja de modelar esto como grafo (en vez de una tabla plana miRNA×gen): el paso de
mensajes deja que `gene` agregue contexto de sus vecinos `cell` (vía `expresses`) y
`gene` (vía `coexpressed_with`) — el embedding final de `g1` no es solo "existe", incorpora
expresión y co-expresión. Esa es la promesa arquitectónica que el paper pone a prueba.

## 4. Por qué el muestreo uniforme de negativos infla el AUROC — el mecanismo

`gene_degree` (`training/eval_topology_baseline.py:100`) puntúa `score(m, g) = deg(g)` —
ignora por completo quién es `m`; la misma fila de puntuaciones se repite para cada miRNA.

Extiendo el juguete con un segundo gen popular, `g5`, regulado por otros tres miRNAs
(`m4, m5, m6`, no `m1`), para poder ilustrar el pareo por grado:

```
deg(g1)=3   deg(g2)=1   deg(g3)=0   deg(g4)=1   deg(g5)=3
```

**Con negativos uniformes** (el muestreador elige cualquier gen no regulado por `m1`, sin
mirar el grado — candidatos `{g3, g4, g5}`):

| Positivo | score deg(g) | Negativo (uniforme) | score deg(g) | ¿gana el positivo? |
|---|:--:|---|:--:|:--:|
| (m1, g1) | 3 | (m1, g4) | 1 | sí |
| (m1, g2) | 1 | (m1, g3) | 0 | sí |

En un grafo real, la mayoría de los genes tienen grado bajo (distribución muy sesgada,
típica de redes biológicas), así que un negativo elegido al azar casi siempre tiene menos
grado que el gen positivo. `deg(g)` gana casi gratis — de ahí que `gene_degree` obtenga
**0.8712 AUROC** con negativos uniformes en los datos reales: no es una heurística lista,
es aritmética de una distribución sesgada.

**Con negativos pareados por grado** (`training/splits.py:202`, exige el mismo bin de
grado que el positivo):

| Positivo | score deg(g) | Negativo (pareado) | score deg(g) | ¿gana el positivo? |
|---|:--:|---|:--:|:--:|
| (m1, g1), grado 3 | 3 | (m1, g5), grado 3 también | 3 | **empate** |
| (m1, g2), grado 1 | 1 | (m1, g4), grado 1 también | 1 | **empate** |

Al forzar que el negativo tenga el mismo grado, `deg(g)` deja de discriminar — empate por
construcción, que en AUROC cuenta como 0.5 (moneda al aire). De ahí que `gene_degree` caiga
a **0.5126** con negativos pareados.

**El mismo atajo lo aprende la red entrenada, no solo la heurística de mano:** un HGT
entrenado con negativos uniformes recibe el mismo gradiente fácil que premia "¿`g` tiene
grado alto? → di positivo". Resultado:
- evaluado con negativos uniformes: **0.8096** (apenas por debajo del heurístico 0.8712 —
  no lo superó)
- evaluado con negativos pareados por grado: **0.5352** (colapsa junto con `gene_degree`,
  0.5126)

Esta es la evidencia directa de que el modelo entrenado con negativos uniformes aprendió
sobre todo el atajo de popularidad. Y es la razón detrás de la regla del proyecto de que el
muestreador de negativos debe ser **el mismo en entrenamiento y en evaluación**
(`PROTOCOLO_Y_CIFRAS_VIGENTES.md`, regla 3) — cambiarlo solo al evaluar no mide dificultad,
mide un desajuste (la "trampa del desajuste" / *mismatch trap*).

## 5. Siguiente paso — la arquitectura del cabezal de predicción (`TargetPredictor`)

Para completar el cuadro falta ver **cómo** el modelo pasa de los embeddings de nodo a un
score de arista, y por qué esa arquitectura no puede expresar afirmaciones específicas de
tipo celular (Contribution 3 de `EVALUATION_AUDIT.md`).

`models/layers.py:77`, clase `TargetPredictor`:

```python
class TargetPredictor(nn.Module):
    """
    Input: concatenated embeddings [miRNA_emb || gene_emb]
    Output: scalar logit per pair
    """
    def __init__(self, hidden_channels: int):
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, mirna_emb, gene_emb):
        return self.mlp(torch.cat([mirna_emb, gene_emb], dim=-1)).squeeze(-1)
```

Puntos clave a explicar en Methods:
- El cabezal toma **solo** `[emb_miRNA ‖ emb_gene]` concatenados (`hidden_channels * 2`
  de entrada) y los pasa por un MLP de 2 capas hasta un logit escalar. **No recibe ningún
  embedding de `cell`.**
- Consecuencia arquitectónica (no corregible reentrenando): para un par `(m, g)` dado, el
  score es el mismo sin importar en qué tipo celular se esté "evaluando" — no existe una
  entrada por la cual la célula pueda modular el score. `analysis/interpret.py:301` puntúa
  cada par miRNA→gen **una sola vez, globalmente**, y luego filtra esa única lista global
  por los miRNAs más salientes de cada tipo celular — la especificidad vive en el filtro,
  nunca en el score.
- Generaliza a un chequeo de tres preguntas para cualquier afirmación de predicción
  específica de contexto: (1) ¿la variable de contexto está en el artefacto/grafo? (2) ¿la
  recibe el cabezal que puntúa? (3) ¿la evaluación varía con ella? Aquí: (1) sí (`cell`
  existe como nodo), (2) no (`TargetPredictor` no la recibe), (3) no puede variar si (2) es
  no.

**Próximos temas pendientes de conversar** (marcar aquí conforme se cubran):
- [x] Nodo / arista / grado / matriz de adyacencia (§1–3)
- [x] Por qué el muestreo uniforme infla el AUROC — mecanismo con ejemplo numérico (§4)
- [x] Arquitectura de `TargetPredictor` y por qué no puede ser específica de tipo celular (§5)
- [ ] Cómo se ve la matriz de adyacencia real en tamaño (cuántos nodos `miRNA`/`gene`/`cell`
      hay realmente, y qué tan dispersa es)
- [ ] Convertir esta explicación en prosa para Methods → "Data and graph construction" y
      "Baselines" en `manuscript.tex`
