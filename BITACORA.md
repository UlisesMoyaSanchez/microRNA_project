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

**El Objective del Abstract** solo decía *qué* se auditó, no *para qué*. Se agregó una
cláusula de cierre: *"...across ten cases spanning eight distinct graphs, to help
correct how link prediction is evaluated on biomedical interaction graphs."*

Se descartó una primera propuesta ("ayudar a que se usen los protocolos correctos en
datos de grafos en el ambiente salud") por los mismos dos motivos que ya aplicaban a
"Protocol" en el bullet 5: (1) "protocolos correctos" es exactamente el overreach que
la nota del bullet 1 (2026-09-07) ya prohibió — el paper dice "corrected", nunca
reclama poseer "el" protocolo correcto. La versión final usa "correct" como **verbo**
sobre la práctica de evaluación, no como adjetivo sobre "protocolos". (2) alcance: solo
se auditaron ocho grafos de interacción biomédica, todos de link prediction —
"grafos en el ambiente salud" en general reclamaría cobertura (grafos de imagenología,
de historia clínica, etc.) que este paper nunca tocó.

Documentado con nota fechada justo después de `\end{abstract}`.

**El mismo día, se acortó más.** El autor pidió quitar la oración de contexto que
abría el Objective ("GNN papers predicting microRNA--target interactions routinely
report AUROC in the 0.91--0.99 range without a model-free baseline..."), dejando
solo la oración del objetivo propiamente dicho. Se hizo, con un ajuste obligatorio:
"this subfield's evaluation practice" se quedaba sin antecedente al desaparecer la
oración que nombraba el subcampo, así que se reescribió a "GNN link-prediction
evaluation practice" (autocontenida). Objective final: *"We audit whether GNN
link-prediction evaluation practice inflates reported performance, across ten cases
spanning eight distinct graphs, to help correct how link prediction is evaluated on
biomedical interaction graphs."*

Costo aceptado a sabiendas: "model-free baseline" ya no se define en el Abstract (se
usa en Methods y Conclusion sin explicarse ahí); sigue definido en el cuerpo del
paper (Resultados/Discusión). Documentado en la misma nota fechada de `main.tex`.
Compila limpio (43 pp., 0 refs indefinidas).

**Methods y Results, reordenados para bajar el peso de "nuestro grafo".** El autor
señaló dos problemas: (1) el Methods abría con la auditoría de 21 papers como si
fuera el método principal, cuando el diseño empírico (otros datos, otros modelos)
es lo que de verdad mide la inflación; y (2) el grid de seis modelos (highlight 3) y
el control de cell-typing aparecían en Results sin haber sido descritos nunca en
Methods.

Se reescribió Methods para abrir con la comparación multi-grafo/multi-modelo (los
diez casos de ocho grafos, nuestro grafo como uno más de la lista, no el sujeto
principal), y se agregó la mención que faltaba de los cinco modelos adicionales y
el control de cell-typing. Se quitó la auditoría de 21 papers de Methods por
completo, y también la oración correspondiente en Results ("Six of 21 papers report
an untrained comparator..."), a petición del autor — la survey le parece secundaria
frente a la comparación empírica.

Costo aceptado a sabiendas, mismo patrón que con "model-free baseline": la oración
del Conclusion *"...and this literature does not check for it"* sigue ahí sin
ningún respaldo metodológico dentro del Abstract (la auditoría de 21 papers ya no
se menciona en Methods ni Results). Sigue siendo cierta y respaldada en el cuerpo
del paper — el hueco es solo dentro del propio Abstract, señalado pero no resuelto
por decisión del autor.

**Acrónimos del Abstract, definidos en su primer uso.** GNN, OGB, GCN, MLP y AUROC
no tenían su forma larga en ningún lado del Abstract (algunos, como AUROC/OGB/GCN/
MLP, tampoco se expanden en el cuerpo del paper). Se agregó la definición completa
la primera vez que aparece cada uno dentro del Abstract; las repeticiones
posteriores quedan en sigla. "RNA-seq" se dejó sin expandir -- terminología
biológica estándar para la audiencia de JBI.

Compila limpio (43 pp., 0 refs indefinidas).

**Conclusion, revisado.** Tres ajustes:

1. "not of any one architecture" → "not of any one model", y en Results "the
   inflation recurs across six architectures" → "...six models" — para que
   highlight 3, Results y Conclusion digan lo mismo. Queda un tercer uso de
   "architecture" sin tocar, en la apertura de Results ("Retraining one
   architecture under both protocols on six graphs..."), que corresponde al
   claim del highlight 1 ("our model") y no se cambió por no haber sido pedido
   explícitamente.
2. Se quitó "and this literature does not check for it" del Conclusion —
   dependía de la auditoría de 21 papers, que ya no aparece en ninguna parte
   del Abstract.
3. El "reporting standard" del Conclusion solo tenía 2 de los 3 componentes
   protegidos por la nota de highlight 5 (le faltaba "matched negative
   sampling", pre-existente desde antes de esta sesión). Se completó para
   que coincida con el highlight 5 y con Discusión (`main.tex:1038-1044`).

Compila limpio (43 pp., 0 refs indefinidas).

**Deudas abiertas al cierre del 2026-09-09** (todas señaladas durante la sesión,
ninguna resuelta todavía):

- Results todavía dice "Retraining one **architecture** under both protocols on six
  graphs..." (línea ~123) — corresponde al claim de highlight 1 ("our model"), no al
  grid de seis modelos, y no se cambió a "model" por no haber sido pedido
  explícitamente. Confirmar si debe unificarse también.
- La etiqueta *"HGT (project model)"* en la Tabla S1 sigue invitando la lectura
  equivocada que su propio caption ya corrige (deuda desde 2026-09-07).
- `training/slurm_heldout_grid.sh:65` hace `cat` del archivo equivocado en modo
  transductivo -- falso fallo tras un éxito real (deuda desde 2026-09-07, es un bug
  de código, no de manuscrito).
- La auditoría de 21 papers sigue sin un highlight propio (deuda desde 2026-09-07),
  y ahora además desapareció de Methods/Results del Abstract -- su único rastro en
  el Abstract ya es indirecto, vía el checklist del Conclusion.

---

## 2026-09-09 (cont.) — La Introducción, acercada al título

El primer párrafo de la Introducción (`main.tex:270-291`) abría con la motivación
biológica y los hallazgos del audit de 21 papers, pero nunca nombraba "protocol"
como término sustantivo (solo aparecía entre paréntesis, como referencia cruzada a
Methods), y la idea de probar sobre varios grafos -- no solo el nuestro -- tampoco
aparecía hasta el párrafo 2. El autor señaló que el párrafo 1 debía sonar más cerca
del título ("The Protocol, Not the Model") desde el principio.

Se cerró el párrafo 1 con: *"This paper asks whether the inflated numbers are a
property of the evaluation protocol, not of any one model or dataset -- and tests
that question not on a single graph, but across eight distinct graphs."* La frase
hace eco deliberado de dos decisiones ya tomadas hoy en el Abstract: "not of any
one model or dataset" es la misma frase del Conclusion, y "eight distinct graphs"
es la cifra del Objective -- para que el lector llegue al párrafo 2 (el detalle
metodológico) ya sabiendo hacia dónde va el argumento.

Compila limpio (44 pp. -- subió una por el párrafo nuevo --, 0 refs indefinidas).

**El mismo día, más recorte.** El autor sintió que, otra vez, el párrafo 1 le daba
demasiado peso al audit de 21 papers frente al diseño empírico -- el mismo patrón ya
corregido ayer en Methods del Abstract. Midiendo: el bloque de cifras del audit
("15 report no model-free baseline... six... 1.7 and 7.4 AUROC points... 17 of the
21...") ocupaba ~100 de las ~250 palabras del párrafo, casi el 40%, en una sola
oración con tres sub-cláusulas. Esas mismas cifras, con nombres de paper, ya están
completas en Related Work (`main.tex:335-341`) -- el propio texto ya prometía "full
results in Related Work below" entre paréntesis, así que no se perdía nada al
recortar aquí.

Se condensó a una oración: *"A literature audit of these 21 papers found that most
omit a model-free baseline -- a simple, no-learning scorer such as node degree, used
as a sanity-check floor -- or default to negatives drawn from unlabeled pairs
without discussion (protocol in Methods, full results in Related Work below)."* Se
conservó la definición breve de "model-free baseline" (primera explicación real del
término en todo el cuerpo del paper, ya que también se quitó del Abstract ayer) pero
se soltaron las cifras específicas. Bloque de ~100 palabras bajó a ~55.

Compila limpio (43 pp., 0 refs indefinidas).

**El mismo día, se pidieron citas para "Neither omission is novel: both are already
documented pitfalls... reviewed below".** Al revisar Related Work
(`main.tex:308-323`) para encontrarlas, resultó que esa sección solo respalda **una**
de las dos mitades: "default to negatives drawn from unlabeled pairs" sí tiene citas
verificadas ya en uso (`kotnis2017analysis, aiyappa2024implicit` para el sesgo de
negative sampling en link prediction en general; `yilmaz2025biasaware` para network
biology específicamente). "Omit a model-free baseline" **no tiene ninguna cita en
todo el paper** que la respalde como pitfall ya documentado en la literatura general
de ML -- no se inventó una (memoria: citas siempre contra fuente primaria, nunca
inventadas -- [[feedback_citation_rigor]]).

Se agregaron las tres citas verificadas a la mitad de negative sampling únicamente;
la mitad de "model-free baseline" queda sin cita inline, a decisión del autor
(opción 1 de dos ofrecidas). La oración "both are already documented pitfalls"
sigue tal cual -- no se reescribió para reflejar que solo una mitad está citada
directamente aquí; sigue siendo cierta en el sentido amplio de "reviewed below",
solo que no las dos con la misma cita inline.

Compila limpio (43 pp., 0 refs indefinidas).

**El mismo día, un tercer ajuste al cierre del párrafo 1.** El autor notó que la
oración nombraba los dos ejes ("not of any one **model or dataset**") pero solo
operaba uno: "tests that question... across eight distinct graphs" -- eso es solo
el eje de los datos (highlight 1: seis grafos, un modelo). El eje del modelo
(highlight 3: seis modelos, un grafo) quedaba prometido al inicio de la oración
pero nunca desarrollado.

Se cerró explícitamente con los dos: *"...and tests that question two ways: across
eight distinct graphs, and across six independently trained models on our own
graph."* Ahora el párrafo 1 anticipa los mismos dos experimentos que ya reflejan
los highlights 1 y 3, y que el párrafo 2 ya desarrollaba ("recurs across
independently trained architectures" -- ese sí cubría ambos ejes desde antes).

Compila limpio (43 pp., 0 refs indefinidas).

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
