# Beamer del proyecto miRNA-MS

Presentacion modular en LaTeX Beamer, 16:9, en espanol.

**Retitulada el 2026-07-27** (decision D1, opcion (a)): el titulo ya no afirma
descubrimiento de redes reguladoras, ni que el trabajo sea sobre esclerosis multiple.
La EM es *procedencia de los datos* y se menciona una sola vez. El titulo anterior
—"Descubrimiento de Redes de Regulacion de miRNAs en Esclerosis Multiple"— afirmaba
dos cosas que el pipeline no sostiene.

## Estructura

- `main.tex`: punto de entrada.
- `sections/01_context.tex`: problema, objetivos y narrativa cientifica.
- `sections/01b_puente_clinico.tex`: puente conceptual para audiencia clinica.
- `sections/02_data_graph.tex`: datos, ejemplos de entrada, preprocessing y grafo.
- `sections/03_model_results.tex`: arquitectura y resultados **de la defensa**.
  Abre con un aviso: presenta el 0.9836 como logro, y esa lectura no sobrevivio.
  Se conserva a proposito como registro de lo que se creia entonces.
- `sections/04_outlook.tex`: los cuatro experimentos de la auditoria.
- `sections/05_estado_actual.tex`: **estado actual — tiene la ultima palabra sobre
  cualquier cifra del deck.** Grafo corregido, 4 semillas, y el resultado entre
  arquitecturas. Usa overlays de beamer (`\onslide<n->`) para construir el argumento
  paso a paso; es la unica seccion animada.

## Orden de precedencia de las cifras

`05_estado_actual.tex` > `04_outlook.tex` > `03_model_results.tex`. Si dos secciones
se contradicen, gana la mas reciente — y el deck lo dice explicitamente en el aviso
que abre la seccion 03, para que nadie lea una cifra vieja como conclusion.

Cifras vigentes (conjunto de **prueba**, grafo `graphs_v3fixed`):

| | |
|---|---|
| Protocolo original (aristas vistas + controles al azar) | 0.9867 +/- 0.0011 (n=4) |
| **Protocolo honesto** (retenidas + pareados) | **0.6276 +/- 0.0070** (n=4) |
| Mejor heuristica sin aprendizaje (`adamic_adar`) | 0.5912 |
| Clasificacion de tipo celular | 0.9916 |

Ojo: `0.6467` y `0.9950` son numeros de **validacion** y estuvieron en el deck por
error hasta 2026-07-27. No volver a usarlos — el propio audit documenta que reportar
el val en vez del test infla +0.020, en un trabajo sobre evaluacion sesgada.

## Figuras

Las figuras de la seccion 05 las **genera un script**, no se dibujan a mano:

```bash
python analysis/make_slide_figures.py     # desde la raiz del repo
```

Lee solo artefactos versionados de `results/comparison/` y escribe
`results/figures/slides_0{1,2,3}_*.pdf` (vector, mas un `.png` para revisar a ojo).
Si un numero cambia, se regenera la figura; **no se edita el PDF ni se teclea el
numero en el .tex**. La paleta del script esta validada (separacion CVD y contraste
computados, no estimados) — no sustituir colores sin re-validar.

Las figuras antiguas de `results/figures/0{1..5}_*.pdf` vienen del pipeline de
analisis (`analysis/slurm_analysis.sh`) y no se editan a mano.

## Compilacion

```bash
cd presentation/beamer_ms_project
pdflatex main.tex && pdflatex main.tex   # dos pasadas: indice y overlays
```

Resultado actual: 79 paginas / 61 frames.
