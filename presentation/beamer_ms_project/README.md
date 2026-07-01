# Beamer del proyecto miRNA-MS

Esta carpeta contiene una presentacion modular en LaTeX Beamer pensada para seguir creciendo.

Estructura:
- `main.tex`: punto de entrada.
- `sections/01_context.tex`: problema, objetivos y narrativa cientifica.
- `sections/02_data_graph.tex`: datos, ejemplos de entrada, preprocessing y grafo.
- `sections/03_model_results.tex`: arquitectura, entrenamiento, resultados y figuras.
- `sections/04_outlook.tex`: limitaciones, trabajo futuro y roadmap de actualizacion.

Notas de mantenimiento:
- Las figuras se referencian desde `../../results/figures/`.
- Si cambian metricas o archivos de resultados, actualizar primero las tablas de `sections/03_model_results.tex`.
- Si aparecen nuevos datasets, ampliar las tablas y ejemplos de `sections/02_data_graph.tex`.

Compilacion futura cuando se necesite:
```bash
cd presentation/beamer_ms_project
pdflatex main.tex
```

No se ha compilado automaticamente en esta tarea.