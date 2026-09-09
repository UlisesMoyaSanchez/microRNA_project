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

## 2026-09-09 — Trim de highlights 3 y 5, y por qué "standard" no se volvió "protocol"

El bullet 3 de los highlights medía 93 caracteres contra el límite de 85 de Elsevier
(deuda abierta desde el 2026-09-07, junto con el bullet 5 que sigue en 142). Se
reescribió de *"Six architectures, our graph: all inflate by 0.24--0.31 AUROC; an
untrained control does not."* a *"Six models tested on our graph data: AUROC
inflates 0.24--0.31, except untrained."* — 81 caracteres.

Dos cambios de vocabulario pedidos por el autor: "architectures" → "models", y "our
graph" → "our graph data". Ninguno de los dos rompe las invariantes ya documentadas
para este bullet: "models" incluso calza mejor con cómo `main.tex` ya nombra este eje
("bullet 3 varía el MODELO a datos fijos"). Se descartó una redacción alternativa
propuesta primero ("probamos 6 modelos adicionalmente y en todos se infla el
resultado") por dos motivos: HGT es uno de los seis, no un séptimo añadido a otros
seis; y "en todos se infla" borra al control sin entrenar, que es justo la excepción
que muestra que la inflación es un artefacto del protocolo de entrenamiento y no algo
que cualquier modelo al azar mostraría — perderlo tergiversa el hallazgo. También se
aclaró de paso que 0.24--0.31 es la **inflación** de AUROC (un delta entre
protocolos), no un AUROC absoluto: los seis modelos tienen AUROCs absolutos muy
distintos (0.92 a 0.49).

Documentado en una nota fechada dentro de `main.tex` junto al resto de notas del
bloque de highlights, para que una edición futura no reintroduzca ninguno de los
dos errores. `c32ae79`

**Bullet 5** medía 142 caracteres — el más largo de los cinco, y la otra mitad de la
misma deuda del 2026-09-07. Se reescribió de *"We propose a minimum reporting
standard: a leak-free split check, a model-free baseline with its margin stated, and
matched negative sampling."* a *"Standard: leak-free splits, margin-stated
model-free baselines, matched negatives."* — 82 caracteres. Se perdió la palabra
"check" (ahora son "splits" a secas), pero se conservaron los tres componentes que
el comentario del código ya protegía, y en particular "model-free" y
"margin-stated" intactos: la Tabla 5 (`tab:correct_protocol`) exige un baseline
específicamente *sin aprendizaje*, no cualquier baseline más débil, y ese
calificador es tan central como el margen declarado.

Se consideró y descartó cambiar "Standard" por "Protocol" para hacer eco del
título ("The Protocol, Not the Model"). No son el mismo concepto: en el resto del
paper "protocol" nombra lo que se audita (el conventional/corrected que colapsa el
AUROC — Tabla 6 `protocol_grid`, Tabla 7, y ahora el propio bullet 1), mientras que
este bullet describe el checklist con el que se audita cualquier protocolo. Usar
"Protocol" aquí habría chocado con el bullet 1 y se acercaba a la advertencia ya
escrita contra "the correct protocol" (2026-09-07): reclamar tener "el protocolo"
es el tipo de sobre-alcance que el paper evita. "Standard" queda como estaba.

Ambos trims compilan limpio (43 pp., 0 refs indefinidas) y quedan documentados con
notas fechadas en `main.tex` junto al resto de notas del bloque de highlights.

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

**Sincronización.** El DGX estaba nueve commits atrás, no tres — le faltaban también los
seis del 2026-09-06. Fast-forward limpio. El artefacto del job 6465 se commiteó desde el
DGX, que es donde se produjo: sin él, la frase de `EVALUATION_AUDIT.md` *"every number
traces to a job ID and a JSON artifact"* habría sido falsa justo para el job que sostiene
la afirmación nueva. `f662e62`

**El bullet 1, reescrito una segunda vez.** Quedó *"Correcting the protocol drops our model
from 0.88--0.99 to 0.62--0.64 on six graphs."* El quiasmo del mismo día se abandonó a
propósito. Tres cosas salieron de la discusión, y valen más que la redacción:

1. **La palabra "protocol" no aparecía en ninguno de los cinco highlights**, mientras el
   título es *The Protocol, Not the Model*. El título y los highlights estaban
   desalineados y nadie lo había notado.
2. **La mala lectura es real y está medida.** Al proponer una reescritura, la primera
   formulación fue *"six previous models plus our model"* — es decir, que se corrieron los
   modelos de los papers auditados. Si quien mejor conoce el paper lo lee así, un revisor
   también. Lo que previene esa lectura es la palabra *our*, no la palabra *architecture*,
   que era lo que se estaba defendiendo por inercia.
3. **"One architecture" es una limitación, no una virtud**, y un highlight no es donde van
   las limitaciones. La generalidad entre arquitecturas ya la carga el bullet 3.

**Qué se replicó de los papers auditados, y qué no** (quedó claro al discutir el punto 2 y
conviene no volver a preguntarlo):

- **Su proceso: no, y deliberadamente no.** Se entrena *nuestra* arquitectura sobre *sus
  grafos*, "under a protocol we control rather than each paper's own" (`main.tex:539`). Sus
  modelos nunca corrieron. El manifiesto de CKSNP-GNN lo dice sin rodeos: *"the code is not
  used or trusted, only the data"*.
- **Su dato: sí, con procedencia pineada.** `sha256` y commit por matriz, y forma y número
  de positivos verificados contra lo que cada paper reporta
  (`hmdd_survey_protocol_verification.json`).
- **Su número headline: nunca se intentó reproducir.** Se toma como reportado. El paper no
  afirma en ninguna forma haberlo reproducido ni haberlo visto fallar.
- **Su piso: sí replica, y es la réplica de verdad.** Reportan comparadores sin entrenar en
  0.8960, 0.8565 y 0.9190; nuestras heurísticas, por otro método sobre las mismas matrices,
  dan 0.8842 y 0.8924. Eso es lo que hace interpretables sus propios márgenes.
- **OGB: reproducción exacta**, pero del leaderboard público, no del modelo de un paper.

Un revisor va a preguntar "¿reprodujeron sus números?". La respuesta es no, está declarada
al abrir §`sec:generalization`, y es defendible porque la pregunta que hace el paper es
otra. Por eso ningún highlight debe insinuar lo contrario.

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
