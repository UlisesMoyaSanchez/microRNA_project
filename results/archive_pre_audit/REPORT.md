# Reporte de Resultados — miRNA-MS Project
**Fecha:** 30 de mayo de 2026  
**Modelo:** `miRNAGraphTransformer` V2 — `checkpoints_v2/best_model.pt` (epoch 191)  
**Hardware:** 4× NVIDIA A100-SXM4-80GB (dgxa100jal)

---

## ¿Qué hace la red neuronal? — Explicación para clínicos

### En una frase
La red aprende a leer simultáneamente la expresión génica de miles de células y los patrones de regulación de miRNAs para **identificar qué tipo celular es cada célula** y **predecir qué genes están siendo controlados por cada miRNA**.

### Analogía médica
Imagina que tienes en tu mano una biopsia de tejido cerebral de un paciente con EM. En lugar de ver las células al microscopio, tienes un perfil molecular de cada célula: cuánto RNA produce para cada uno de 3,000 genes. La red neuronal actúa como un patólogo computacional que:

1. **Clasifica** cada célula en su tipo (linfocito Th17, microglía, oligodendrocito, etc.)
2. **Evalúa** qué miRNAs están "apretando o aflojando el volumen" de cada gen en cada tipo celular
3. **Puntúa** cada par miRNA→gen con una probabilidad de que esa regulación sea real y relevante

### Entradas del modelo

| Entrada | Descripción clínica | Ejemplo concreto |
|---------|--------------------|--------------------|
| Expresión génica por célula | Perfil de actividad de 3,000 genes en cada una de las 110,798 células (scRNA-seq) | Célula #4721: IL17A ↑↑, FOXP3 ↓, RORC ↑ → probablemente Th17 |
| Expresión de miRNAs | Niveles de 2,460 miRNAs humanos medidos en sangre/tejido (miRNA-seq, GEO) | hsa-miR-146a-3p: expresión elevada en muestras MS vs. control |
| Relaciones miRNA→gen conocidas | 44,186 interacciones validadas experimentalmente (miRTarBase, miRDB) | hsa-miR-23a-3p está registrado como regulador de CCL7 |

> Los datos provienen de muestras de sangre periférica y LCR de **pacientes con EM y controles sanos** (bases GEO: GSE107742, GSE119453, GSE289530, GSE41995 + CellxGene).

### Salidas del modelo

| Salida | Significado clínico | Ejemplo de resultado |
|--------|--------------------|-----------------------|
| **Tipo celular predicho** (1 de 11 clases) | Etiqueta automática de cada célula sin necesidad de marcadores | "Esta célula es un oligodendrocito" — exactitud 99.4 % |
| **Score de regulación** (0 → 1) | Probabilidad de que ese miRNA controle activamente ese gen *en el contexto de EM* | hsa-miR-23a-3p → CCL7: **0.986** (muy alta confianza) |
| **Saliencia por tipo celular** | Ranking de miRNAs más influyentes para la identidad de cada tipo celular | hsa-miR-140-5p aparece en top-10 de 6 tipos celulares distintos |

### ¿Por qué una red de grafos y no un modelo clásico?
Los modelos clásicos (regresión, Random Forest) analizan cada célula o cada gen de forma independiente. Una **red de grafos heterogénea** conecta al mismo tiempo células, genes y miRNAs como nodos de una red, y aprende de los *patrones relacionales*: si el miRNA A regula los genes B y C, y esos genes se co-expresan en células Th17 de pacientes con EM, la red detecta ese circuito completo de manera emergente — algo imposible con análisis univariante.

---

## 1. Comparativa actualizada con baselines y ablaciones

> Evaluación consolidada del job `baselines_5042.out` en modo `EVALUATE ONLY`, usando el modelo V2 ya entrenado y comparándolo contra modelos clásicos y ablaciones estructurales.

| Modelo | val_loss | AUROC | AUPRC | cell_acc | cell_f1 | Lectura |
|--------|:--------:|:-----:|:-----:|:--------:|:-------:|---------|
| **HGT V2** | **0.1541** | **0.9836** | **0.9731** | **0.9961** | **0.9935** | Mejor modelo completo: único con link prediction competitivo y clasificación casi perfecta |
| Random | 1.4245 | nan | nan | 0.0909 | 0.0799 | Referencia negativa |
| MLP | 0.0992 | nan | nan | 0.9254 | 0.8883 | Clasificación aceptable, pero muy por debajo del grafo completo |
| Homo GCN | 0.1411 | nan | nan | 0.8945 | 0.8631 | El uso de grafo homogéneo no recupera la estructura multi-tipo del problema |
| Ablation no miRNA | 0.0159 | nan | nan | 0.9895 | 0.9821 | La tarea de clasificación celular sigue siendo fuerte aun sin nodos miRNA |
| Ablation no coexpr | 0.0118 | nan | nan | 0.9933 | 0.9889 | La clasificación se conserva casi intacta sin el bloque de coexpresión |

**Conclusión de la comparativa:** el resultado nuevo deja claro que la señal de clasificación celular es más fácil que la de regulación. Por eso las ablaciones pueden mantener `cell_acc` alta, pero no ofrecen AUROC/AUPRC para link prediction. En términos de objetivo científico, **V2 sigue siendo el mejor modelo integral**, porque combina clasificación casi perfecta con capacidad real para puntuar enlaces miRNA→gen.

**Lectura metodológica:** los nuevos baselines muestran que quitar miRNAs o quitar coexpresión afecta menos la clasificación de tipos celulares de lo que afecta la riqueza biológica del modelo. Esto refuerza que el valor añadido del HGT no es solo etiquetar células, sino priorizar circuitos reguladores en contexto.

---

## 2. Métricas en Test Set (modelo V2)

> Evaluación sobre el 10% de células reservado como test set (n ≈ 11,080 células).

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **AUROC** | **0.9776 – 0.9794** | Predicción de enlaces miRNA→gen (link prediction) |
| **AUPRC** | **0.9673 – 0.9728** | Área bajo curva Precision-Recall (robusto con clases desbalanceadas) |
| **Cell Accuracy** | **0.9941 – 0.9943** | Clasificación de tipo celular (11 clases) |
| **Cell F1-macro** | **0.9890 – 0.9892** | F1 promediado igualitariamente entre tipos celulares |

Variación entre jobs (5034–5037) debida al muestreo aleatorio del test loader; diferencias < 0.002 en todas las métricas.

---

## 3. Grafo Heterogéneo

```
HeteroData(
  gene     { x=[3000, 1]    }      # 3,000 genes altamente variables
  cell     { x=[110798, 50], y=[110798] }  # 110,798 células (11 tipos)
  miRNA    { x=[2460, 64]   }      # 2,460 miRNAs humanos (miRDB v6.0 ≥80)
  (miRNA, regulates, gene)         edge_index=[2, 44186]   # miRTarBase validado
  (gene, regulated_by, miRNA)      edge_index=[2, 44186]
  (cell, expresses, gene)          edge_index=[2, 15978902]
  (gene, expressed_in, cell)       edge_index=[2, 15978902]
  (gene, coexpressed_with, gene)   edge_index=[2, 1130]    # PCC ≥ 0.60
)
```

---

## 4. Saliencia de miRNAs por Tipo Celular (Top-10)

> Saliencia = |∂(logit clase c)/∂embedding miRNA_i| promediado sobre 2,000 células
> estratificadas. Un valor alto indica que ese miRNA influye fuertemente en la
> clasificación de ese tipo celular.

### Tipos celulares focales para EM (esclerosis múltiple)

#### Th17
| Rank | miRNA | Saliencia |
|------|-------|-----------|
| 1 | hsa-miR-4659b-3p | 0.3974 |
| 2 | hsa-miR-140-5p | 0.3516 |
| 3 | hsa-miR-514b-3p | 0.1921 |
| 4 | hsa-miR-6872-3p | 0.1770 |
| 5 | hsa-miR-5580-3p | 0.1716 |
| 6 | hsa-miR-202-3p | 0.1314 |
| 7 | hsa-miR-6745 | 0.0946 |
| 8 | **hsa-miR-146a-3p** | **0.0880** |
| 9 | hsa-miR-6833-5p | 0.0857 |
| 10 | hsa-miR-551b-5p | 0.0680 |

> ⚠️ **hsa-miR-146a-3p** es regulador conocido de NF-κB en células Th17 — validación biológica positiva del modelo.

#### Microglia
| Rank | miRNA | Saliencia |
|------|-------|-----------|
| 1 | hsa-miR-3680-3p | 0.3909 |
| 2 | hsa-miR-6745 | 0.3448 |
| 3 | hsa-miR-202-3p | 0.3236 |
| 4 | hsa-miR-5580-3p | 0.2864 |
| 5 | hsa-miR-3180-5p | 0.2624 |
| 6 | hsa-miR-602 | 0.2292 |
| 7 | hsa-miR-520e-3p | 0.1398 |
| 8 | hsa-miR-4659b-3p | 0.1271 |
| 9 | hsa-miR-6872-3p | 0.1197 |
| 10 | hsa-miR-4477b | 0.1127 |

#### Oligodendrocyte
| Rank | miRNA | Saliencia |
|------|-------|-----------|
| 1 | hsa-miR-4659b-3p | 0.7552 |
| 2 | **hsa-miR-140-5p** | **0.6627** |
| 3 | hsa-miR-520e-3p | 0.2129 |
| 4 | hsa-miR-4769-3p | 0.1187 |
| 5 | hsa-miR-4477b | 0.0785 |

> ⚠️ **hsa-miR-140-5p** tiene evidencia publicada en diferenciación de oligodendrocitos — segundo punto de validación biológica.

### Resumen cross-celular: miRNAs pan-MS

| miRNA | Tipos celulares en top-10 | Relevancia |
|-------|--------------------------|------------|
| **hsa-miR-140-5p** | T\_cell, Th17, Treg, NK\_cell, Oligodendrocyte, Monocyte | Candidato maestro regulador en EM |
| **hsa-miR-4659b-3p** | T\_cell, Th17, Treg, NK\_cell, Oligodendrocyte, Monocyte | Alta co-ocurrencia inmune + glial |
| **hsa-miR-6745** | B\_cell, CD4\_T, NK\_cell, T\_cell, Microglia | Regulador pan-linfoide |
| **hsa-miR-202-3p** | Astrocyte, Microglia, Treg, T\_cell | Señal glial-inmune |

---

## 5. Interacciones miRNA→Gen de Alta Confianza

> 44,186 pares miRNA→gen evaluados (miRTarBase validado, score = P(regulación)).

| Umbral de score | N° pares |
|----------------|---------|
| > 0.95 | **167** |
| > 0.90 | 1,245 |
| > 0.80 | 5,379 |

### Top-10 pares más confiables

| miRNA | Gen diana | Score | Relevancia EM |
|-------|-----------|-------|---------------|
| hsa-miR-23a-3p | **CCL7** | 0.986 | Quimiocina; reclutamiento monocitos en EM |
| hsa-miR-23a-3p | **PRDM1** | 0.984 | Factor diferenciación plasmablastos |
| hsa-miR-23c | HOXB5 | 0.982 | Factor transcripción desarrollo |
| hsa-miR-1273h-5p | KCNIP1 | 0.982 | Canal K+; señalización neuronal |
| hsa-miR-23c | **MAP3K1** | 0.981 | Kinasa; activación NF-κB/MAPK |
| hsa-miR-548ae-3p | COL4A1 | 0.981 | Colágeno; barrera hematoencefálica |
| hsa-miR-548ap-3p | RASSF3 | 0.980 | Supresor tumoral; apoptosis |
| hsa-miR-548ae-3p | **IGFBP3** | 0.980 | Modulador IGF; neuroprotección |
| hsa-miR-23a-3p | MAP3K1 | 0.980 | — |
| hsa-miR-23c | ATP11C | 0.980 | Flipasa; homeostasis membrana |

> **hsa-miR-23a-3p → CCL7** es el par más confiable del modelo. CCL7 (MCP-3) es una quimiocina clave en el reclutamiento de monocitos inflamatorios al SNC en EM — relevancia directa al fenotipo estudiado.

---

## 6. Archivos Generados

### Interpretación (`results/interpretation/`)
| Archivo | Tamaño | Contenido |
|---------|--------|-----------|
| `mirna_saliency_by_celltype.tsv` | 612 KB | Matriz 2,460 miRNAs × 11 tipos celulares |
| `all_edge_scores.tsv` | 1.4 MB | 44,186 pares miRNA→gen con score de predicción |
| `top_circuits_by_celltype.tsv` | 87 KB | Top circuitos regulatorios por tipo celular |
| `enrichment/<celltype>_GO.tsv` | — | Enriquecimiento GO Biological Process (job 5038) |
| `enrichment/<celltype>_KEGG.tsv` | — | Enriquecimiento KEGG Pathways (job 5038) |

### Figuras (`results/figures/`)
| Figura | Descripción |
|--------|-------------|
| [01_umap_embeddings.pdf](figures/01_umap_embeddings.pdf) | UMAP de 110,798 células: (a) tipo celular, (b) condición MS/control, (c) saliencia miRNA top |
| [02_mirna_heatmap.pdf](figures/02_mirna_heatmap.pdf) | Heatmap saliencia: top miRNAs × 11 tipos celulares |
| [03_network_Th17.pdf](figures/03_network_Th17.pdf) | Circuito regulatorio miRNA→gen en Th17 |
| [03_network_Microglia.pdf](figures/03_network_Microglia.pdf) | Circuito regulatorio miRNA→gen en Microglia |
| [03_network_CD4_T.pdf](figures/03_network_CD4_T.pdf) | Circuito regulatorio en CD4+ T |
| [03_network_CD8_T.pdf](figures/03_network_CD8_T.pdf) | Circuito regulatorio en CD8+ T |
| [03_network_B_cell.pdf](figures/03_network_B_cell.pdf) | Circuito regulatorio en células B |
| [03_network_Monocyte.pdf](figures/03_network_Monocyte.pdf) | Circuito regulatorio en monocitos |
| [04_top_mirna_barplot.pdf](figures/04_top_mirna_barplot.pdf) | Top-20 miRNAs por saliencia media cross-celular |
| [05_auroc_auprc.pdf](figures/05_auroc_auprc.pdf) | Curvas ROC (AUROC=0.978) y PR (AUPRC=0.970) en test set |

---

## 7. Próximos Pasos Recomendados

1. **Revisar enriquecimiento GO/KEGG** — job 5038 en curso; resultados en `results/interpretation/enrichment/`
2. **Validación externa** — contrastar hsa-miR-23a-3p→CCL7 y hsa-miR-140-5p en Oligodendrocytes con bases públicas (miRTarBase experimental, GEO GSE289530)
3. **Análisis diferencial MS vs. Control** — usar embeddings de células del UMAP segmentados por condición para identificar miRNAs con saliencia específicamente alterada en MS
4. **Validación de ablaciones** — cuantificar de forma explícita qué parte del rendimiento corresponde a clasificación celular y qué parte a link prediction para no sobredimensionar las ablaciones
5. **Fine-tuning adicional** — V2 no alcanzó plateau en epoch 200 (val_loss aún decreciendo); extender a 300 épocas con lr scheduler cosine podría mejorar AUROC >0.99

---

*Generado automáticamente — pipeline: `train.py` → `evaluate.py` → `interpret.py` → `visualize.py`*
