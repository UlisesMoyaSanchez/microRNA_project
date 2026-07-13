> # ⚠️ SUPERSEDED — DO NOT CITE
>
> **Archived 2026-07-12.** Kept as the historical record, not as a result.
>
> **What it claims:** *"rendimiento fuerte de link prediction"* — AUROC **0.9836**, with
> `nan` for every baseline.
>
> **Why it is wrong, twice:**
> 1. The **0.9836** came from an evaluation with no edge-level train/test split: the
>    scored miRNA→gene edges were the same ones supervised during training. Retrained
>    honestly, the same model scores **0.6467** (job 5605).
> 2. The **`nan` baselines were a bug**, not a result (`HeteroData.get()` silently
>    returned `None` for the tuple edge-type key and disabled link prediction). Once
>    fixed, `homo_gcn` reaches 0.9170 and `ablation_no_coexpr` 0.9374 — so the claim
>    that V2 was the *only* model doing link prediction was never true.
>
> **Cell-type classification is unaffected** and remains genuine (0.9950).
>
> **Correct numbers:** [`results/EVALUATION_AUDIT.md`](../EVALUATION_AUDIT.md) ·
> Spanish summary: [`results/RESUMEN_AUDITORIA.md`](../RESUMEN_AUDITORIA.md)

---

# Resumen Ejecutivo — miRNA-MS Project

**Fecha:** 30 de mayo de 2026  
**Referencia de comparativa:** `logs/baselines_5042.out`

## Mensaje central

El proyecto integra transcriptomica de celula unica, expresion de miRNAs e interacciones miRNA-gen en un grafo heterogeneo para estudiar esclerosis multiple con resolucion por tipo celular. El mejor modelo actual, `miRNAGraphTransformer V2`, no solo clasifica con alta precision los tipos celulares, sino que tambien puntua pares miRNA-gen con rendimiento fuerte de link prediction.

## Resultado principal

| Modelo | val_loss | AUROC | AUPRC | cell_acc | cell_f1 |
|--------|:--------:|:-----:|:-----:|:--------:|:-------:|
| **HGT V2** | **0.1541** | **0.9836** | **0.9731** | **0.9961** | **0.9935** |
| Random | 1.4245 | nan | nan | 0.0909 | 0.0799 |
| MLP | 0.0992 | nan | nan | 0.9254 | 0.8883 |
| Homo GCN | 0.1411 | nan | nan | 0.8945 | 0.8631 |
| Ablation no miRNA | 0.0159 | nan | nan | 0.9895 | 0.9821 |
| Ablation no coexpr | 0.0118 | nan | nan | 0.9933 | 0.9889 |

## Que significan las ablaciones

- `Ablation no miRNA` elimina las aristas `miRNA -> gen` del grafo. El modelo sigue viendo las celulas y los genes, pero ya no puede usar evidencia regulatoria directa de miRNAs; por eso esta prueba sirve para medir cuanto aportan los miRNAs a la tarea y a la priorizacion de enlaces.
- `Ablation no coexpr` elimina las aristas `gen -> gen` de coexpresion. En este caso el modelo conserva la informacion de celulas y de miRNA-gen, pero pierde la capa estructural que conecta genes con patrones de coexpresion; asi se aísla el valor de esa redundancia biologica en el grafo.
- En ambos casos la clasificacion celular sigue siendo alta, lo que indica que clasificar tipos celulares es mas facil que inferir circuitos reguladores; la diferencia importante aparece al intentar explicar regulacion biologica de forma mas fina.

## Como se ven los datos de entrada

El pipeline no entra con un solo archivo, sino con tres fuentes principales que luego se convierten en un grafo heterogeneo:

| Entrada | Forma | Ejemplo de contenido |
|---------|-------|----------------------|
| scRNA-seq procesado | `celulas x genes` | una fila por celula, columnas como `condition`, `cell_type` y miles de genes |
| Expresion de miRNAs | `miRNAs x muestras` | filas como `hsa-miR-140-5p`, columnas como `GSM...` y valores normalizados |
| Grafo final | nodos `miRNA`, `gene`, `cell` | aristas `miRNA->gen`, `cell->gene` y `gene->gene` |

Ejemplo simplificado de entrada:

```text
scRNA-seq (una celula):
cell_id    condition   cell_type       MS4A1   PTPRC   HLA-DRA
cell_001   MS          B_cell          1.24    0.00    2.17

miRNA bulk (una muestra):
miRNA             GSM_001   GSM_002   GSM_003
hsa-miR-140-5p     8.41      8.12      8.55
hsa-miR-146a-3p    6.02      6.11      5.94
```

## Es supervisado?

Si. El entrenamiento es multi-tarea y supervisado en dos partes:

- La clasificacion de tipos celulares usa como etiqueta `cell_type`, creada en el preprocesamiento de scRNA-seq a partir de scores de genes marcadores.
- La prediccion de enlaces miRNA-gen usa como positivos las interacciones cargadas desde `data/raw/mirtarbase_hsa.tsv`, que en esta pipeline corresponde a predicciones de miRDB v6.0 filtradas por score alto; el nombre del archivo se conserva por compatibilidad.

En otras palabras, el modelo aprende con etiquetas conocidas, pero esas etiquetas no vienen del mismo lugar: una viene de anotacion celular derivada de marcadores y la otra de una base externa de interacciones miRNA-target.

## Como interpretar esto correctamente

- La clasificacion celular es una tarea mas facil que la prediccion regulatoria.
- Por eso las ablaciones conservan accuracy alta, pero no demuestran capacidad comparable para inferir enlaces miRNA-gen.
- El valor diferencial de `HGT V2` es que combina dos cosas a la vez: clasificacion casi perfecta y capacidad real de priorizar circuitos regulatorios.

## Hallazgos biologicos ya defendibles

- Se identificaron miRNAs con relevancia transversal o focal por tipo celular, incluyendo `hsa-miR-140-5p`, `hsa-miR-4659b-3p` y `hsa-miR-146a-3p`.
- El proyecto ya produce circuitos miRNA-gen priorizados y enriquecimiento funcional por tipo celular.
- Esto permite pasar de una descripcion transcriptomica a hipotesis mecanisticas concretas para EM.

## Riesgos y limitaciones

- La expresion de miRNAs es bulk y no de celula unica.
- Parte del conocimiento miRNA-gen proviene de prediccion computacional.
- Aun falta separar con mas detalle que parte del rendimiento se debe a estructura celular y que parte a regulacion biologica real.

## Siguiente paso recomendado

Construir una validacion externa enfocada en los circuitos top del modelo V2 y presentar por separado la evidencia de clasificacion celular y la evidencia de regulacion miRNA-gen.