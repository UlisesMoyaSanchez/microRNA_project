# Protocolo y cifras vigentes — leer antes de reportar cualquier número

**Última actualización:** 2026-07-29
**Para qué existe:** este proyecto ya reportó una vez un AUROC de 0.9836 como logro del
modelo. No era una invención: era una medición correcta bajo un protocolo que filtraba.
El costo de descubrirlo tarde fue alto. Este archivo es el resguardo para no repetirlo —
es la lista corta de qué se puede reportar, qué no, y cómo comprobarlo.

**Jerarquía documental:** este archivo manda sobre cualquier cifra suelta en slides o
notas. Para el razonamiento completo, `EVALUATION_AUDIT.md`. Para el plan de publicación,
`../goal.md`. Resumen para colaboradores en español: `RESUMEN_AUDITORIA.md`.

---

## 1. Las cinco reglas

1. **Reportar siempre el conjunto de PRUEBA, nunca el de validación.**
   Seleccionar en `val_auroc` y reportarlo infla **+0.020** (0.6467 val → 0.6271 test).
   Los números `0.6467` y `0.9950` estuvieron en el deck por error hasta 2026-07-27: son
   de validación. No volver a usarlos.

2. **Las aristas de evaluación no pueden estar en entrenamiento, en ninguna dirección.**
   Comprobarlo, no suponerlo: `python training/test_edge_split.py --config <cfg>`. Las
   ocho comprobaciones deben dar cero. Sin eso, ningún número de link prediction significa
   nada.

3. **Negativos pareados por grado, y el mismo muestreador en entrenamiento y evaluación.**
   Cambiar el muestreador solo al evaluar mide un desajuste, no la dificultad — es la
   "trampa del desajuste" que casi reportamos (un modelo entrenado con negativos duros da
   *menos* contra negativos uniformes, lo cual es absurdo a primera vista).

4. **Ningún AUROC se reporta sin su control sin aprendizaje al lado.**
   Los pisos vigentes sobre `graphs_v3fixed` están abajo. Si un modelo no le gana a
   `gene_degree`, no hay resultado que reportar.

5. **Las cifras van con su job de SLURM y su artefacto JSON.**
   Si un número no se puede rastrear a un archivo bajo `results/comparison/`, no entra a
   un documento. Y si un número cambia, se regenera la figura — **no se edita el PDF ni se
   teclea el número en el `.tex`**.

---

## 2. Cifras vigentes

Grafo `data/graphs_v3fixed/` (verificado de forma independiente, job 5728). Conjunto de
**prueba**. Aristas miRNA→gen retenidas: 4,418.

### Link prediction — la rejilla 2×2, 4 semillas {123, 777, 2024, 7}

| | controles al azar | controles pareados |
|---|:--:|:--:|
| aristas **vistas** (protocolo original) | **0.9867 ± 0.0011** | 0.9248 ± 0.0045 |
| aristas **retenidas** (protocolo honesto) | 0.8096 ± 0.0059 | **0.6276 ± 0.0070** |

AUPRC del extremo honesto: **0.6598 ± 0.0049**.
Artefactos: `comparison/multiseed_auroc_test_v3fixed.json`,
`multiseed_auprc_test_v3fixed.json`, `multiseed_seen_edges_test_v3fixed.json`.
Jobs: 5743–5768 (entrenamiento), 5808–5844 y 5816–5823 (evaluación).

**Atribución** (celdas con entrenamiento y evaluación pareados):
solo controles −0.062 · solo partición −0.177 · **ambos −0.359** · suma de las dos
por separado −0.239 → **super-aditivo**, así que corregir una sola subestima el daño.

### Pisos sin aprendizaje (job 5827)

| heurística | controles al azar | controles pareados |
|---|:--:|:--:|
| `gene_degree` — ignora al miRNA | **0.8712** | 0.5126 |
| `adamic_adar` — la más fuerte | 0.8630 | **0.5912** |

Artefacto: `comparison/topology_baseline_v3fixed_test.json`.
**Idénticos a los del grafo anterior**, porque `index_maps.pkl` es byte a byte igual entre
`data/graphs/` y `data/graphs_v3fixed/`: las correcciones nunca tocaron la topología
miRNA→gen. Útil como comprobación de que no se movió nada más.

### Entre arquitecturas (n=1, semilla 42, jobs 5849–5852)

| arquitectura | original | honesto | inflación |
|---|:--:|:--:|:--:|
| `hgt_v2` | 0.9222 | 0.6080 | **+0.314** |
| `homo_gcn` | 0.9177 | **0.6236** | **+0.294** |
| `ablation_no_coexpr` | 0.9128 | 0.6185 | **+0.294** |
| `mlp` (sin grafo) | 0.7622 | 0.5202 | **+0.242** |
| `random` (control) | 0.4895 | 0.4938 | −0.004 |

Artefactos: `comparison/comparison_table_checkpoints_v3fixed_baselines_*.tsv`.

**Es n=1 a propósito.** La inflación es 0.24–0.31 y la dispersión entre semillas donde sí
se pudo medir es 0.005–0.02, así que el efecto supera al ruido con holgura. **Reportar
como n=1, nunca con ±.** Si un revisor pide semillas: 16 jobs, ~8 h.

**Ojo con este cuadro:** `hgt_v2` da 0.6080 aquí y 0.6276 arriba. Misma arquitectura y
mismo criterio de selección, pero `run_baselines.py` entrena en una sola GPU con su propio
bucle y queda 0.02–0.06 más bajo. **Esta tabla sirve para comparar arquitecturas entre sí
en condiciones idénticas, no para repetir el titular.**

### Clasificación de tipo celular

**0.9916** (prueba). Real, con partición correcta a nivel de célula, y no afectada por
ninguno de los problemas anteriores. Es la parte del proyecto que siempre se sostuvo.

**Ahora con su control sin aprendizaje al lado (regla 4), job 5853, 2026-07-29:**
sobre las mismas `X_pca` (50 componentes) y el mismo split, sin pasar mensajes por el
grafo, `nearest_centroid` da **0.4654** y `logistic_regression` **0.6692**. La brecha con
0.9916 es de **32 puntos** — muy por encima de los 3.6 puntos que separan al HGT de
`adamic_adar` en link prediction. **No es circular**: si el número fuera solo una
reconstrucción de `argmax(marker score)` vía PCA, un clasificador simple sobre esas mismas
`X_pca` habría cerrado casi toda la brecha, y no lo hace. El paso de mensajes sobre
`expresses`/`coexpressed_with` le da al modelo acceso a información que 50 componentes de
PCA no capturan.

También se comprobó, desde el `.h5ad` procesado, que `cell_type` es exactamente
`argmax(sc.tl.score_genes)` — coincide en el 100 % de las células, incluido el split de
prueba (`preprocess_scrna.py:34-58`). Eso es la definición de la etiqueta, no una fuga: el
modelo nunca ve esos scores, solo `X_pca` y el grafo, y aun así los controles sin grafo se
quedan muy por debajo.

**Advertencia que sí viaja con el número:** la etiqueta `cell_type` nunca se validó contra
una anotación independiente (FACS, atlas de referencia). El control descarta "cualquier
clasificador trivial llega a 0.99", no la pregunta más profunda de si el marker-score
argmax es la verdad biológica. Es una limitación más débil que la que se estaba temiendo,
no una que la elimine.

Artefacto: `comparison/celltype_baseline_config_v2_edgesplit_test.json`. Script:
`training/eval_celltype_baseline.py` (`training/slurm_celltype_baseline.sh`).

---

## 3. Lo que NO se puede reportar

| Cifra | Por qué no |
|---|---|
| `0.9836` / `0.8828` | Protocolo con fuga, n=1, grafo anterior, y **no recomputables** (la ruta con fuga se borró en `8a12ce3`). Sustituidas por 0.9867 ± 0.0011 y 0.9248 ± 0.0045. |
| `0.6467`, `0.9950`, `0.9961` | Son de **validación**, no de prueba. |
| `0.9853`, `0.9766`, `0.9758`, `0.7760`, `0.5150` | Niveles absolutos de los Experimentos 1 y 2, protocolo con fuga. De esos experimentos solo sobrevive la **diferencia** (p. ej. −0.009 al tapar la arista), no el nivel. |
| `0.9374`, `0.9170`, `0.7899` | Comparativa de arquitecturas bajo el protocolo con fuga. Sustituida por la tabla de arriba, **cuya conclusión es la contraria**: el GCN simple le gana al HGT. |
| Cualquier circuito concreto | `miR-23a-3p→CCL7` y compañía están ordenados por un cabezal que supera a una heurística por 3.6 puntos, y **nunca fueron específicos de tipo celular** (ver §4). Validación externa **suspendida**. |
| Saliencia de miRNAs como señal biológica | Las features de miRNA son reproducibles con `torch.randn` (max\|diff\| = 0.0): **no contienen datos**. Un heatmap de saliencia sobre ellas es una imagen de ruido. |
| Cualquier afirmación sobre EM | Decisión **D1 (a)**, 2026-07-27: la etiqueta de enfermedad nunca entra al modelo. La EM es procedencia de los datos y se menciona una vez. |

### Figuras retiradas del deck (2026-07-27)

`results/figures/01_umap_embeddings.pdf`, `02_mirna_heatmap.pdf`,
`03_network_*.pdf`, `04_top_mirna_barplot.pdf`, `05_auroc_auprc.pdf` son de mayo de 2026:
grafo con errores y modelo con fuga. `05_auroc_auprc.pdf` son las curvas ROC del modelo con
fuga; `02_mirna_heatmap.pdf` es saliencia de features sin contenido. **No reutilizar.**
Las figuras vigentes son `slides_0{1,2,3}_*.pdf` y las genera
`analysis/make_slide_figures.py` desde los artefactos versionados.

---

## 4. El hallazgo arquitectónico (no lo arregla reentrenar)

`TargetPredictor` (`models/layers.py:77`) recibe `[emb_miRNA ‖ emb_gene]` y **no recibe la
célula**. `analysis/interpret.py:301` puntúa cada par **una sola vez, globalmente**, y
después filtra esa lista única por los miRNAs más salientes de cada tipo celular.

Consecuencia: **`miR-23a-3p→CCL7` tiene el mismo score en todos los tipos celulares.** La
premisa central del proyecto no llegó a implementarse y con esta arquitectura no podía.

Generalizable, y es el aporte que más viaja: **toda afirmación de predicción específica de
contexto debería comprobarse contra tres preguntas** — (1) ¿la variable de contexto está en
el artefacto? (2) ¿la recibe el cabezal que puntúa? (3) ¿la evaluación varía con ella?
Aquí fallamos las tres.

---

## 5. Cómo reproducir lo vigente

```bash
# Puerta de entrada: si la partición filtra, nada de lo demás significa algo. Solo CPU.
python training/test_edge_split.py --config configs/config_v3fixed_edgesplit.yaml

# Piso sin aprendizaje sobre el grafo corregido           -> job 5827
python training/eval_topology_baseline.py \
    --config configs/config_v3fixed_edgesplit.yaml --split test \
    --out results/comparison/topology_baseline_v3fixed_test.json

# Rejilla multi-semilla (una corrida por semilla y condición) -> 5808-5844
sbatch --export=ALL,CONFIG=configs/config_v3fixed_edgesplit_s123.yaml,\
CKPT=checkpoints_v3fixed_edgesplit_s123/best_model.pt,SPLIT=test \
    training/slurm_heldout_grid.sh
python training/aggregate_seeds.py --split test --metric auroc \
    --checkpoint-prefix checkpoints_v3fixed \
    --topology-baseline results/comparison/topology_baseline_v3fixed_test.json \
    --out results/comparison/multiseed_auroc_test_v3fixed.json

# Rejilla entre arquitecturas (una corrida por celda)      -> 5849-5852
for c in edgesplit edgesplit_uniform transductive transductive_uniform; do
    sbatch --export=ALL,CONFIG=configs/config_v3fixed_baselines_${c}.yaml \
        training/slurm_baselines.sh
done

# Control sin aprendizaje para clasificación celular       -> job 5853
sbatch --export=ALL,CONFIG=configs/config_v2_edgesplit.yaml,SPLIT=test \
    training/slurm_celltype_baseline.sh

# Figuras del deck (nunca se editan a mano)
python analysis/make_slide_figures.py
```

**Dos trampas de infraestructura ya conocidas:** varias herramientas escriben a nombres de
archivo **sin identidad de grafo** (`eval_topology_baseline.py --out` y
`aggregate_seeds.py --out` por defecto), así que se pasa `--out` explícito o se sobrescribe
el registro anterior. Y un job de SLURM puede terminar en verde sin haber producido nada
(`set -o pipefail` bajo `dash`, `GDAL_DATA` sin definir): **comprobar que el archivo existe,
no que el job salió con 0.**

---

## 6. La lección que se nos aplicó a nosotros

El criterio de selección por `val_loss` ya estaba identificado y corregido en `train.py`
(ahí está documentado que guardaba un modelo al azar, 0.5324, para uno que alcanza 0.6268).
**El arreglo nunca llegó a `run_baselines.py`**, y se detectó por un síntoma —paro temprano
en la época 29 en vez de 124— con la rejilla ya corriendo. Hubo que cancelarla y repetirla.

La tabla sin corregir habría estado mal **en la dirección que favorece nuestra propia
tesis**. Es exactamente el tipo de error que este trabajo denuncia, encontrado en nuestra
propia herramienta de auditoría. Vale más decirlo en la discusión que un párrafo genérico
de limitaciones.
