# Guía de auditoría independiente

**Proyecto:** *The Protocol, Not the Model: Evaluation Bias Inflates Link Prediction on Biomedical Interaction Graphs* (manuscrito JBI, en preparación).
**Escrita:** 2026-10-02. **Commit base al escribirla:** `01c58f6` (rama `feat/protocheck`, no pusheada).
**Pendiente antes de enviarla a los auditores** (los mantenedores completan): commit de congelamiento `________`, contacto `________`, fecha límite `________`, acceso al grafo propio (§3).

---

## 0. Qué se pide y por qué

Hicimos una auditoría de cómo se evalúa la predicción de enlaces en grafos biomédicos y construimos una herramienta (`protocheck`) que advierte de los errores que describimos. **Quienes escribimos esto somos los menos indicados para saber si está bien.** Esta guía pide que otras personas rehagan el trabajo *sin ver primero nuestras respuestas*, y que intenten romperlo.

Lo que se audita son **cinco afirmaciones** (§1). Para cada una hay una pista de trabajo (§4) con su método, su regla de decisión y las condiciones bajo las cuales **estaríamos equivocados**. Un resultado que contradiga el paper es un resultado útil, no un problema: se reporta tal cual.

**Lo que esta auditoría NO evalúa:** la validez biológica de ninguna predicción; si el número publicado de cada paper de la encuesta es exacto (nunca lo reprodujimos, §6); ni la calidad de la escritura.

---

## 1. Las afirmaciones

| ID | Afirmación (como la hace el manuscrito) | Pista |
|---|---|---|
| **C1** | Reentrenando **una** arquitectura bajo los dos protocolos en **seis grafos** (el nuestro y cinco de papers de la encuesta), el AUROC colapsa de 0.879–0.992 a 0.618–0.641. | A, B |
| **C2** | En nuestro grafo el protocolo convencional da 0.9867 ± 0.0011 y el corregido 0.6276 ± 0.0070, a 3.6 puntos de una heurística sin aprendizaje (0.5912). La inflación es **super-aditiva** (corregir solo una de las dos fallas subestima el daño) y reaparece en seis modelos (+0.24 a +0.31 AUROC) salvo en un control sin entrenar. | B |
| **C3** | En los seis grafos de C1, las heurísticas sin aprendizaje (grado, vecinos comunes, Adamic–Adar…) dan un piso alto bajo el protocolo convencional y ~0.55–0.59 bajo el corregido. Sobre dos benchmarks OGB reproducen el leaderboard. | A |
| **C4** | En una encuesta de 21 papers de GNN, la mayoría no reporta un baseline sin aprendizaje; los que sí lo reportan le ganan por pocos puntos; y casi todos validan sobre aristas. Los detalles de cada dimensión se dan en la pista D, no aquí, para no anclar. | D |
| **C5** | `protocheck` detecta los cuatro problemas que dice detectar, no dispara sobre un protocolo corregido, y sus límites son los declarados. | E |

Las cifras de C1–C3 vienen del Abstract del manuscrito. **No se entregan** nuestros artefactos intermedios hasta la Fase 3 (§2).

---

## 2. Cómo evitar que nuestras respuestas contaminen la suya

Fases, en orden:

| Fase | Qué ocurre | Qué ven los auditores |
|---|---|---|
| **0. Pre-registro** | Se acuerdan por escrito las reglas de decisión de §4 (tolerancias, qué cuenta como "no reproducido"). **Se pueden endurecer, no relajar, después de ver datos.** | Esta guía |
| **1. Implementación ciega** | Cada auditor implementa su propio código desde la especificación de §4 y obtiene números. | Guía, datos crudos (§3), codebook ciego, plantilla en blanco |
| **2. Entrega** | Entregan su reporte (§7) **antes** de ver nuestros resultados. | — |
| **3. Comparación y apertura** | Se entregan nuestros artefactos y código; se comparan; las discrepancias se discuten con ambas partes. | Todo |
| **4. Reporte final** | Reporte con las discrepancias resueltas y las que no. | Todo |

**Se retiene hasta la Fase 3:** `results/` (cifras y artefactos), `BITACORA.md`, `results/EVALUATION_AUDIT.md`, `results/literature_survey*.tsv` (nuestras calificaciones), `training/eval_*.py` y `protocheck/` (nuestras implementaciones; leerlas antes de implementar la pista A o B anula la independencia). Quien haga la pista **E** necesita `protocheck/`: idealmente es una persona distinta de quien haga A/B; si es la misma, E va al final.

**Pista D:** no leer §2.2 ni la Tabla 3 del manuscrito antes de calificar (contienen nuestros conteos).

---

## 3. Materiales y procedencia de los datos

`data/raw/` está en `.gitignore`: **las matrices no viajan con el repositorio**. Eso es una ventaja para la independencia: se obtienen de las fuentes originales y se verifican con hash. Los scripts `data/01_download/download_hmdd_survey_*.py` documentan cómo las obtuvimos (léanse en Fase 3, o úsense solo los hashes de abajo). **Los comandos exactos de descarga y los hashes verificados están en [`DATA_ACCESS.md`](DATA_ACCESS.md).**

| Grafo | Papers de la encuesta que lo usan | Fuente | Forma · positivos |
|---|---|---|---|
| `canonical5430` | MGCNSS, NIMGSA, HLGNN-MDA (**comparten una sola matriz**) | `15136943622/MGCNSS@f8a87e69` (`data/miRNA_disease_matrix.csv`), `zhanglabNKU/NIMGSA@5bf10a93` (`m-d.txt`), `LiangYu-Xidian/HLGNN-MDA@e3ac8240` (`.../1.miRNA-disease associations/association.txt`) | 495 × 383 · 5,430 |
| `cksnp_gnn` | CKSNP-GNN | `http://public.aibiochem.net/DNA_RNA/Genes_Human-miRNA-disease-Associations/code.zip`, sha256 `3277b666…2952`. **Ojo:** el README/código del zip describe otro modelo (GAEMDA); solo se usa la matriz | 901 × 877 · 16,427 |
| `digamn` | DiGAMN | `yinboliu-git/GAMN@6695f4e4`, `data/MDA-CF/m_d.csv`, sha256 `354f0aa1…7688` | 917 × 792 · 14,550 |
| `meahne` | MEAHNE | `yyx-hc/MEAHNE@96f3faa6` | 1,296 × 11,783 · 17,972 |
| `couplemda` | CoupleMDA | `lizhj39/CoupleMDA@c9128cb7` (listas de pares train/test) | 2,090 × 1,754 · 13,509 train + 1,523 test tras quitar 43 pares solapados |
| `ogbl-ddi`, `ogbl-ppa` | — | paquete `ogb` (`PygLinkPropPredDataset`), split oficial | ddi: 4,267 nodos |
| **grafo propio** | — | cohorte de scRNA-seq + miRDB v6.0; **~550 MB, no versionado** | 2,460 miRNAs × 3,000 genes · 44,186 aristas |

**Verificación que pedimos como primer paso (C3 indirecta):** que las tres matrices de `canonical5430` sean idénticas (SHA-256 de los valores parseados, `78f1a3cd7a8b62cf6341208f2c651f732e7a28f391d93125d46f8fb3aa77d04e`), y que forma y número de positivos coincidan con la tabla de arriba. Si no coinciden, **detenerse y avisar**.

**Grafo propio:** C2 y la mitad de C1 dependen de él. Se entrega bajo solicitud, con su SHA-256 (los mantenedores lo calculan al congelar). Quien no lo obtenga puede auditar C1 solo sobre los cinco grafos externos y debe declararlo.

**Software:** Python con numpy, scipy, scikit-learn; PyTorch (+ PyG si se desea) solo para entrenar modelos en la pista B. Las pistas A, D y E corren sin GPU.

---

## 4. Las pistas

Todas las tolerancias son una **propuesta nuestra** para fijar en la Fase 0.

### Pista A — Piso sin aprendizaje (C3)

**Objetivo:** reimplementar las heurísticas y medirlas bajo la rejilla 2×2, de forma independiente.

**Especificación.** Matriz de adyacencia binaria `A` (filas = miRNA, columnas = enfermedad/gen) construida solo con las aristas **visibles** para la heurística. Grados `deg_row`, `deg_col` sobre `A`.

- `gene_degree(m,g) = deg_col[g]` (ignora la fila).
- `pref_attach(m,g) = deg_row[m]·deg_col[g]`.
- `common_neigh(m,g) = Σ_{m'≠m} |N(m)∩N(m')| · A[m',g]`, es decir `(S·A)` con `S = A Aᵀ` y diagonal en cero.
- `adamic_adar(m,g)`: igual, pesando cada columna compartida por `1/log(max(deg_col,2))`.
- (Grafos no dirigidos como OGB: simetrizar, quitar auto-lazos; las fórmulas son las clásicas, más `resource_alloc` con peso `1/deg`.)

**Rejilla 2×2.** Aristas **vistas** (la heurística ve las aristas evaluadas: el protocolo convencional) frente a **retenidas** (no las ve); negativos **uniformes** frente a **emparejados por grado**:

- División aleatoria de los positivos; fracción de prueba 0.1 (0.2 para MEAHNE); razón de negativos 1:1. Pueden elegir otras si las declaran y usan las mismas en todas las celdas.
- Negativo uniforme: para cada positivo `(m,g)`, un `(m,g')` con `g'` uniforme y `(m,g')` no positivo conocido.
- Negativo emparejado: `g'` en el **mismo contenedor de grado** que `g`. Contenedores log-espaciados (8) sobre `log(1+grado)`, grado calculado **solo con aristas de entrenamiento**; si el contenedor no ofrece candidato válido tras varios intentos, caer a uniforme y reportar la tasa de caída.
- Métrica: AUROC de positivos frente a negativos (empates promediados).
- Todos los positivos conocidos (entrenamiento + prueba) se excluyen como negativos.

**Qué reportar:** AUROC por heurística, por celda, por grafo, con ≥4 particiones y su desviación.

**Regla de decisión (propuesta).** C3 se considera **reproducida** si, **en cada uno de los seis grafos de C1** (no en OGB, donde las heurísticas de vecindad son fuertes aun con negativos emparejados), la mejor heurística de la celda (retenidas, emparejados) queda por debajo de 0.65 y la de (vistas, uniformes) por encima de 0.85. Compararán sus cifras con las nuestras en Fase 3: la discrepancia es aceptable si `|Δ| ≤ 0.015` o `|z| ≤ 2` (Welch, medias de particiones). **Estaríamos equivocados** si una implementación correcta da un piso emparejado ≥ 0.75 en algún grafo externo, o uno uniforme < 0.7 en un grafo con muchas columnas muertas.

**Celda determinista (OGB):** en `ogbl-ddi`, con el grafo visible `dataset[0].edge_index` simetrizado (2,135,822 aristas dirigidas), los 133,489 positivos de validación y los 101,882 negativos oficiales de validación, no hay aleatoriedad: sus AUROC por heurística deben coincidir con los nuestros con `|Δ| ≤ 1e-3`. Es la prueba más dura de la especificación.

### Pista B — Reentrenar bajo ambos protocolos (C1, C2)

**Objetivo:** comprobar el *fenómeno*, no nuestra arquitectura. Cada auditor **elige su propio modelo** (una GCN homogénea, un MLP con embeddings, un transformer; lo que prefiera) y lo entrena sobre los grafos de §3 bajo las cuatro celdas de la rejilla, con **el mismo muestreador de negativos en entrenamiento y evaluación** dentro de cada celda.

**Trampas que ya conocemos y que hay que evitar (o reproducir a propósito):**
1. *Fuga por relación inversa:* quitar la arista de prueba de la relación directa pero dejarla en la inversa.
2. *Trampa del desajuste:* entrenar con un muestreador y evaluar con otro mide el desajuste, no la dificultad. Un modelo entrenado con negativos duros puntúa *menos* contra negativos uniformes.
3. *Reportar validación como prueba* infla +0.02 en nuestro caso.
4. *Seleccionar el mejor época sobre prueba.*

**Qué reportar:** AUROC en las 4 celdas por grafo, ≥3 semillas, la partición y el criterio de selección declarados, más el control sin entrenar (pesos aleatorios) y el piso de la pista A.

**Regla de decisión (propuesta).** C1 se considera **reproducida en el fenómeno** si, en **cada** grafo, (i) AUROC(vistas, uniformes) − AUROC(retenidas, emparejados) ≥ 0.15, y (ii) el AUROC corregido queda a ≤ 0.10 del piso corregido de la pista A. C2 (super-aditividad) se comprueba con las celdas apareadas: `caída(ambas) > caída(solo negativos) + caída(solo partición)` en magnitud; el control sin entrenar no debe moverse más que su ruido. **Estaríamos equivocados** si un modelo razonable bien implementado conserva ≥ 0.85 de AUROC bajo el protocolo corregido en algún grafo, o si la caída en algún grafo es < 0.05.

**Advertencia importante para nuestro grafo (Tabla S1).** El número 0.6276 sale de entrenar con 4 GPUs (DDP) con un `train_mask` completo cargado en cada rank sin `DistributedSampler`; la misma arquitectura en 1 GPU da 0.6080. **Comparen dentro de un mismo régimen, nunca entre tablas.** Por eso C2 se prueba con *su* propio modelo y no exigiendo 0.6276 exacto.

### Pista C — Mecanismo (opcional, 1 persona de A o B)

Reproducir con un ejemplo mínimo (grafo pequeño sintético) que (a) la fuga por arista de prueba visible permite a un modelo sin parámetros entrenados recuperar la arista, y (b) el desajuste de muestreadores produce el resultado absurdo descrito. Sirve para decidir si los diagramas de Methods (Figuras 1 y 2) son correctos.

### Pista D — Encuesta de literatura (C4)

**Tarea:** calificar de forma **independiente y ciega** los 21 papers de `survey_rating_template.tsv` con las reglas de `SURVEY_CODEBOOK_blind.md` (cuatro dimensiones: validación sobre aristas, aristas de prueba removidas del grafo del encoder, procedencia de los negativos, baseline sin aprendizaje en la tabla de resultados). Reglas duras del codebook: **fuente primaria únicamente** (texto completo del paper, no su código ni un paper que lo cite); **cada `yes`/`no` lleva una cita textual**; `unclear` es una afirmación sobre el paper, no sobre su esfuerzo.

**Importante:** un `unclear` es una respuesta legítima y frecuente; no lo resuelvan por inferencia. Si no pueden acceder al texto completo, dejen la fila en blanco (no `unclear`).

**Comparación (Fase 3).** Acuerdo crudo y kappa de Cohen por dimensión contra nuestra adjudicación, **más** la lista de discrepancias por paper. Nuestro propio acuerdo entre dos calificadores humanos fue de 0.71 a 0.95 de acuerdo crudo según la dimensión, con kappa de −0.05 a 0.49; en las dimensiones donde casi todas las respuestas caen en una categoría el kappa se degenera por prevalencia, así que **reporten el acuerdo crudo junto al kappa**. Una discrepancia no se resuelve por mayoría: la resuelve una tercera lectura de la fuente primaria con cita.

**Regla de decisión (propuesta).** C4 se considera **reproducida** si, tras resolver las discrepancias contra la fuente, los conteos de cada dimensión difieren de los nuestros en ≤ 2 papers y la dirección de la conclusión no cambia (la mayoría no reporta baseline sin aprendizaje; los que lo reportan lo superan por pocos puntos). **Estaríamos equivocados** si ≥ 5 de nuestras calificaciones cambian al leer la fuente, o si la mayoría de los papers sí reporta un baseline sin aprendizaje en su tabla de resultados.

### Pista E — Poner a prueba `protocheck` (C5)

**Instalación:** `git clone <repo> && cd <repo> && git checkout <commit de congelamiento> && pip install -r protocheck/requirements.txt`. Luego `pytest tests/test_protocheck.py` (17 tests) y `python -m protocheck.examples.make_examples` para generar cinco conjuntos sintéticos. Léase `protocheck/README.md`.

**Tareas, en orden:**
1. **Caja negra.** Con los ejemplos sintéticos, ¿cada uno dispara exactamente su advertencia? ¿El `clean` no dispara ninguna?
2. **Formatos reales.** Pasar los grafos de §3 por el CLI en al menos dos formatos de entrada (pares de aristas y matriz + pares de prueba). ¿El resultado coincide con lo que la pista A calculó por su cuenta?
3. **Adversarial.** Construyan conjuntos que *deberían* disparar cada check pero de una forma que no anticipamos, y conjuntos que **no** deberían disparar pero sí lo hacen. Ideas (no exhaustivas): aristas duplicadas o auto-lazos; la fuga por la relación inversa en el modo bipartito; negativos emparejados con contenedores calculados sobre **todas** las aristas (no solo entrenamiento); fuga a través de *features* de nodo derivadas de las etiquetas de prueba (la herramienta declara que no la detecta: ¿cuán fácil es hacerlo sin querer?); un grafo con columnas muertas pero una razón de negativos distinta; casos con muy pocos positivos de prueba; índices en base 1 frente a base 0.
4. **Umbrales.** Los umbrales por defecto (`protocheck.DEFAULT_THRESHOLDS`) son heurísticos. ¿Hay un grafo realista donde produzcan un veredicto claramente absurdo? Reporten el número, no solo el veredicto.
5. **Paridad.** Comparar los scorers de `protocheck` contra su propia implementación de la pista A, en `ogbl-ddi` (celda determinista).

**Regla de decisión (propuesta).** C5 se considera **reproducida** si pasan 1 y 2, la paridad de 5 es ≤ 1e-3, y los hallazgos de 3–4 se reportan y clasifican. Una falla de detección real (un conjunto con fuga que la herramienta marca `ok`) es un **hallazgo de severidad alta** aunque la herramienta ya lo declare en su nota de alcance, si el caso es plausible en la práctica.

---

## 5. Condiciones que invalidan una pista

Detengan y avisen (no sigan "a ver qué sale") si: las matrices de §3 no coinciden con forma/positivos/hash; sus heurísticas dan menos de 0.5 en la celda sin aprendizaje con negativos uniformes (probable error de signo o de índices); su modelo entrenado alcanza AUROC 1.0 bajo el protocolo corregido (probable fuga propia); o el control sin entrenar supera 0.6 bajo el protocolo corregido.

---

## 6. Lo que ya sabemos que es débil (para que lo ataquen primero)

Ordenado de lo que más nos preocupa a lo que menos. Declararlo aquí no sustituye ponerlo a prueba.

1. **Los números de los papers de la encuesta nunca se reprodujeron.** En la encuesta se toman "como reportados". Los reentrenamientos de seis grafos usan *nuestra* arquitectura sobre *sus* datos, así que demuestran que el protocolo convencional infla *un* modelo entrenado, no que la cifra de cada paper esté inflada. El manuscrito lo dice en la Discusión; la Introducción lo afirmaba de más hasta el 2026-10-02.
2. **C1 usa una sola arquitectura** (la nuestra) sobre los seis grafos. La generalidad entre modelos solo se probó en el grafo propio.
3. **El confound DDP/1-GPU** (Tabla S1, §4 pista B): el grid de seis modelos no es el modelo del titular. Lo que sobrevive es la comparación *dentro* de cada régimen.
4. **Anomalía abierta (Tabla 7).** El margen de discriminación de nuestro grafo se *cierra* bajo el protocolo corregido mientras que el de los cinco grafos externos se *abre*. Probamos que no es sobreajuste (está presente en los seis) y no tenemos explicación. Es la parte menos entendida del trabajo.
5. **Encuesta:** dos calificadores humanos, acuerdo moderado en dos dimensiones, n = 21; en una dimensión (aristas de prueba removidas del grafo del encoder) 11 de 21 quedan `unclear`. Un titular previo ("0/22 sin baseline") resultó falso y se corrigió tras una segunda calificación ciega.
6. **Etiquetas de tipo celular:** `cell_type` es el `argmax` de un puntaje de marcadores; nunca se validó contra una anotación independiente. El control descarta "cualquier clasificador trivial llega a 0.99", no la pregunta biológica.
7. **`protocheck`:** los umbrales son heurísticos; su validación es casi tautológica para fuga y negativos (compara conjuntos de aristas e histogramas de grado: valida la implementación, no un descubrimiento); `dead_candidates` re-deriva una estadística que ya calculábamos y **no está validado de forma independiente**; en la comparación con nuestro pipeline en torch, el criterio fijado de antemano ("dentro de 2 sd") **falló en 2 de 5 grafos** y añadimos una comparación de Welch después de verlo (diferencia máxima de 0.011 AUROC); `ogbl-ppa` no se probó.
8. **El grafo propio no está versionado** (§3): su reproducción depende de que los autores lo entreguen.

---

## 7. Qué devolver

Un archivo `AUDIT_REPORT_<nombre>.md` por auditor, **antes de la Fase 3**, con:

```markdown
# Reporte de auditoría — <nombre>, <fecha>
## Entorno
Commit/versión usados: …  Python/paquetes: …  Hardware: …  Tiempo dedicado por pista: …
## Por afirmación (C1…C5)
| ID | Estado | Evidencia (números, archivos) | Desviaciones del protocolo |
Estado ∈ {reproducida | reproducida con discrepancia | no reproducida | no se pudo probar}
## Decisiones y desviaciones
Todo lo que cambió respecto a esta guía y por qué (tolerancias, particiones, modelos).
## Hallazgos
| # | Severidad (alta/media/baja) | Dónde | Qué pasa | Cómo reproducirlo |
## Lo que no pude probar
```

Adjunten el código con el que obtuvieron cada número, las semillas, y los archivos de salida. **Los números sin código no cuentan.** Si algo les pareció confuso en esta guía, díganlo: una ambigüedad en la especificación es un hallazgo.

**Qué hacemos con lo que devuelvan:** cada discrepancia o hallazgo se registra en la bitácora del proyecto con su resolución, y los cambios al manuscrito citan el reporte que los motivó. Si una afirmación no se reproduce, se corrige o se retira; no se reformula para que parezca que sí.
