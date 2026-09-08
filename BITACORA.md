# Bitácora — miRNA-MS Project

Registro cronológico de decisiones y hallazgos. Entradas nuevas arriba.

**Qué va aquí:** decisiones que costaron trabajo tomar, hipótesis que se probaron y
fallaron, confounds encontrados, y cambios de alcance del paper. Una entrada por día de
trabajo, no por commit.

**Qué NO va aquí,** y dónde vive en su lugar:

| Contenido | Documento |
|---|---|
| Las cifras vigentes y sus reglas | [`results/PROTOCOLO_Y_CIFRAS_VIGENTES.md`](results/PROTOCOLO_Y_CIFRAS_VIGENTES.md) |
| Los hallazgos canónicos, con job IDs y artefactos | [`results/EVALUATION_AUDIT.md`](results/EVALUATION_AUDIT.md) |
| El plan y qué falta para someter | [`goal.md`](goal.md) §3 |
| Por qué el manuscrito dice lo que dice | comentarios `%% STAGE:` en `manuscript/jbi/main.tex` |
| El detalle de cada cambio de código | `git log` |

> **Nota sobre las entradas anteriores al 2026-09-07.** Están reconstruidas a partir de
> `git log` y de las notas fechadas en `goal.md` y `EVALUATION_AUDIT.md`, no escritas en su
> momento. Los hechos y las fechas son fiables; el razonamiento de cada día, solo hasta
> donde esos documentos lo registraron.

---

## 2026-09-07 — Un confound de régimen de entrenamiento bajo el grid de arquitecturas

**Título y highlights.** El título pasó a *"The Protocol, Not the Model: Evaluation Bias
Inflates Link Prediction on Biomedical Interaction Graphs"* — la conclusión del propio
paper. Se pierde "microRNA" del título, sostenido ahora por el keyword, la primera frase
del abstract y los highlights. Los highlights 1 y 2 se fusionaron (el 2 no aportaba ningún
hecho que el 1 no cubriera ya) y el slot liberado fue al estándar de reporte propuesto, que
no tenía bullet. Los bullets 1 y 3 se reescribieron como quiasmo — *"Six graphs, one
architecture"* / *"Six architectures, our graph"* — porque se leían como el mismo resultado
dos veces siendo los dos ejes del claim de generalidad. `9b00a6a`

**El hallazgo del día.** La Tabla 1 y la Tabla S1 reportan el mismo modelo, sobre el mismo
grafo, bajo la misma configuración, y difieren en **0.065 AUROC** en la celda convencional
(0.9867 contra 0.9222). Descartado en orden:

1. **El split** — hipótesis inicial, y era falsa. El job 6465 evalúa el checkpoint del
   headline en validación: 0.98586 contra 0.98582 en test, sobre los mismos 44,186 edges.
   Diferencia de 0.00004.
2. **El seed** — los cuatro seeds del headline abarcan 0.0024. El hueco es 0.065.
3. **El presupuesto** — ambos convergieron, early stopping en 196 y 181, ambos
   seleccionando sobre `val_auroc`.

**La causa es el driver.** El headline sale de `train.py` bajo DDP a 4 GPUs; el grid sale
de `run_baselines.py` a 1 GPU. Bajo DDP los cuatro ranks cargan **todo** `train_mask`
(`train.py:348`, no hay `DistributedSampler`) con un seed por rank (`train.py:314`), así que
un paso del optimizador consume 4×512 celdas en vez de 512. Mejor `val_auroc` 0.9952 contra
0.9345.

**Por qué nadie lo vio en seis semanas:** los YAML son idénticos salvo seed y
`checkpoint_dir`, y el comentario del config lo audita explícitamente y acierta. El confound
no está en el config — está en el script de SLURM. Lección general: un config que se
autocertifica como controlado no cubre lo que el scheduler le hace encima.

**Qué sobrevive.** Los dos claims de highlights. La inflación de seis arquitecturas es
internamente controlada (un driver, una GPU, un seed, un split para las seis filas), y el
colapso de seis grafos también (los dos brazos de protocolo de cada grafo se entrenan igual
entre sí). Lo que no sobrevive es cualquier lectura cruzada entre tablas, incluida la
afirmación secundaria de que una GCN homogénea le gana al transformer bajo el protocolo
corregido: ese transformer no es la arquitectura como el proyecto la entrena de verdad.

Declarado en Methods, Results, los captions de Fig. 2 y Tabla S1 —en el generador además
del archivo generado, para que una regeneración no lo borre— y registrado en el audit
canónico. `0d88ae6`

**Deudas abiertas:** dos highlights exceden el límite de 85 caracteres de Elsevier; la
auditoría de 21 papers sigue sin bullet propio; la etiqueta `HGT (project model)` en la
Tabla S1 sigue invitando la lectura equivocada que su caption ahora corrige;
`slurm_heldout_grid.sh:65` hace `cat` del nombre de archivo equivocado en modo transductivo
(falso fallo tras un éxito real).

---

## 2026-09-06 — La encuesta, doble-calificada contra las fuentes primarias

La encuesta de literatura se calificó por segunda vez a ciegas contra los papers originales.
El titular *"0 de 22 reportan un baseline model-free"* resultó **falso**: 6 de 21 sí lo
hacen, y lo despejan por 1.7–7.4 puntos sin comentarlo — un hallazgo más interesante que el
original, porque el problema deja de ser que nadie mire el piso y pasa a ser que nadie lea
su propio margen. La encuesta es n=21, no 22. Se adjudicaron los 14 desacuerdos entre
calificadores contra la fuente primaria, se le dio a D2 un valor de "no aplica", y se barrió
el repo por claims que las correcciones no habían alcanzado. Nuestro grafo bajó a "uno de
seis". El draft de BMC se retiró y el generador se repuntó al manuscrito vivo.
`69fa688`…`b42abd3`

---

## 2026-08-31 — Los Resultados, reordenados de lo general a lo particular

Los Resultados se reordenaron para abrir con el colapso de seis grafos en vez de con
nuestro grafo, y se añadió la figura del colapso cross-graph. El título se ensanchó a
"biomedical interaction graphs" al notar que el objeto anterior —miRNA–target— era el más
estrecho del paper: cinco de los seis grafos del claim principal son miRNA–enfermedad y dos
benchmarks son OGB. `dfa31fc`…`c487b71`

---

## 2026-08-22 — La hipótesis de densidad, refutada

La auditoría de topología sobre 7 papers de HMDD encontró que solo hay **5 grafos
distintos** (tres papers comparten una matriz). La hipótesis de que la densidad explicaba la
inflación se probó y **falló**; el mecanismo real son las columnas candidatas muertas. Se
añadieron el grid de protocolo model-free sobre todos los grafos, los grids entrenados para
los cinco grafos de la encuesta, y la Tabla 7 (el margen de discriminación bajo cada
protocolo). `0b8e511`…`d7cf13a`

---

## 2026-08-06 → 08-13 — Manuscrito JBI y validación externa

Arranca el draft de JBI. Se añade la auditoría ligera de OGB (baselines model-free sobre
ogbl-ddi y ogbl-ppa, donde las heurísticas reproducen el leaderboard exactamente) y el
spot-check de la encuesta de literatura. `3a43983`…`9c2707d`

---

## 2026-07-27 → 07-29 — Re-medición sobre el grafo arreglado

Todas las cifras headline se re-midieron sobre `data/graphs_v3fixed/` con 4 seeds. **Cada
claim de la auditoría sobrevive**, dentro de 1σ de su contraparte pre-fix — lo que es en sí
evidencia de que el hallazgo es sobre el *protocolo* y no sobre un grafo con bugs. Se
añadieron las cuatro celdas del grid cross-arquitectura (jobs 5849–5852; ver la entrada del
2026-09-07 sobre cómo se entrenaron), se arregló la selección de modelo en
`run_baselines.py` —seleccionar por `val_loss` guardaba un "mejor modelo" en nivel de azar—
y se añadió el control sin aprendizaje para clasificación de tipo celular.
`2c9fe06`…`a031c14`

---

## 2026-07-16 → 07-21 — Dos bugs de construcción de grafo

Se encontraron y arreglaron dos bugs (selección de genes por co-expresión, umbral de
expresión célula→gen), naciendo `graphs_v3fixed`. El protocolo transductivo se restauró
detrás de un flag explícito `edge_split: false` para poder re-medir la fila de referencia
"edges seen" sobre el grafo nuevo. Se hicieron *capaces de fallar* las dos verificaciones de
grafo — antes no podían. Se pre-registró el submuestreo pareado por densidad y se
registraron las cinco decisiones MS abiertas. `17d7dc7`…`b47999d`

---

## Antes del 2026-07-16

91 commits desde el 2026-05-28. La auditoría original corrió el 2026-07-13 y es la que
convirtió el proyecto de "un modelo de predicción de enlaces" en "una crítica de método":
ver `results/EVALUATION_AUDIT.md`, que la documenta entera con job IDs.
