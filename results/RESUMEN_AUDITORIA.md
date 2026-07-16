# Resumen de la auditoría de resultados — Proyecto miRNA-MS

**Para:** colaboradores clínicos del proyecto.
**Fecha:** 13 de julio de 2026. **Todas las cifras son del conjunto de PRUEBA**
(4,418 interacciones que el modelo nunca vio, ni para entrenar ni para elegir el mejor modelo).
**Documento técnico completo (inglés):** [`EVALUATION_AUDIT.md`](EVALUATION_AUDIT.md)

---

## Lo que hay que saber, en tres frases

1. **El resultado principal que les presentamos era un artefacto.** El AUROC de **0.9836**
   para la predicción de regulación miRNA→gen no sobrevive a una evaluación correcta. El
   número honesto es **0.6271**.
2. **La clasificación de tipos celulares sí es real** y no cambia: exactitud **0.9916**. Esa
   parte del trabajo está intacta.
3. **Lo encontramos nosotros, no un revisor.** Esa es la razón de ser de esta auditoría, y
   es la diferencia entre corregir un análisis y retractar un artículo.

---

## Qué salió mal, en términos clínicos

**Estábamos validando el score en los mismos pacientes que lo generaron.**

La partición de los datos separaba *células* en entrenamiento y prueba — eso estaba bien.
Pero las 44,186 interacciones miRNA→gen que usábamos para **evaluar** eran **las mismas** que
habíamos usado para **entrenar**. Nunca apartamos un conjunto de interacciones que el modelo
no hubiera visto nunca.

Es exactamente el error de construir un score de riesgo con una cohorte y luego "validarlo"
en esa misma cohorte. El desempeño sale inflado por construcción.

Al apartar el **20% de las interacciones** (8,836 pares que el modelo no ve nunca: ni como
respuesta, ni dibujadas en el grafo de entrada, ni en dirección inversa) y reentrenar, el
AUROC cae de **0.9836 a 0.6271**.

---

## El hallazgo más importante: los controles no estaban pareados

Este es el punto que conviene entender bien, porque es el que convierte el error en un
resultado publicable.

Para medir si el modelo acierta, se le presentan **pares verdaderos** (casos) y **pares
falsos** (controles). Nosotros elegíamos los controles **al azar**.

El problema: existe un **factor de confusión** con nombre propio — la **popularidad del gen**.
Algunos genes son diana de muchísimos miRNAs. Un control elegido al azar casi siempre cae en
un gen impopular, así que distinguirlo de un par verdadero es trivial.

**Lo comprobamos con un control sin modelo.** Construimos un "marcador tonto" que **ignora
por completo al miRNA** y solo pregunta: *¿qué tan popular es este gen?*

| "Prueba diagnóstica" | Controles al azar | Controles **pareados** por popularidad |
|---|:--:|:--:|
| Marcador tonto (solo popularidad, sin modelo) | **0.8712** | 0.5126 |
| Adamic-Adar (topología, sin aprendizaje) | 0.8630 | **0.5912** |
| **Nuestro modelo HGT** | 0.8056 | **0.6271** |

Léanlo así:

- **Con controles al azar, un marcador que ni siquiera mira el miRNA alcanza 0.8712.** Casi
  el nivel que presentábamos como logro del modelo. Es decir: **el protocolo de evaluación,
  no el modelo, hacía la mayor parte del trabajo.**
- **Con controles pareados, ese marcador tonto se desploma al azar (0.5126)**, como debe.
- Nuestro modelo aguanta en **0.6271** — por encima del azar, sí, pero apenas **3.6 puntos
  por encima de Adamic-Adar**, una fórmula de dos líneas de 1999 que no aprende nada.

**Conclusión:** el modelo no está infiriendo regulación biológica. Está haciendo un
completado de grafo apenas mejor que trivial.

### Y el dato más contundente de toda la auditoría

Entrenamos el modelo **con controles al azar** (el protocolo original) sobre las aristas
apartadas, para compararlo de forma limpia contra el marcador tonto en igualdad de
condiciones:

| Sobre aristas apartadas, con controles al azar | AUROC |
|---|:--:|
| Popularidad del gen — **sin modelo, sin aprender, ignora el miRNA** | **0.8712** |
| **Nuestro HGT entrenado de punta a punta** | **0.8056** |

**La heurística de una línea le gana al transformer por 6.6 puntos.**

Bajo el protocolo de evaluación que usábamos, el modelo profundo no solo era innecesario:
era **peor** que contar cuántos miRNAs ya apuntan a ese gen.

### Y el mecanismo, atrapado en el acto

A ese mismo modelo (entrenado con controles al azar) le pusimos **controles pareados**.
Resultado: **0.5118** — azar puro, y prácticamente idéntico al 0.5126 del marcador tonto.

**El modelo entrenado con controles al azar se *convirtió* en la heurística de popularidad.**
No aprendió nada más. Los controles mal elegidos no solo halagan al modelo: **seleccionan un
modelo que no ha aprendido nada transferible.**

---

## Un cuarto hallazgo: el modelo nunca fue específico de tipo celular

Esto no es estadístico, es **arquitectónico**, y no se arregla reentrenando.

El cabezal que puntúa las interacciones (`TargetPredictor`) recibe **solo** el miRNA y el gen.
**No recibe la célula.** Y el análisis puntúa cada par miRNA→gen **una sola vez, globalmente**,
y después filtra esa lista única por los miRNAs más salientes de cada tipo celular.

Consecuencia: **`miR-23a-3p → CCL7` tiene exactamente el mismo score en todos los tipos
celulares.** La "especificidad de tipo celular" venía **solo** del filtro de saliencia sobre el
miRNA, nunca del score de la interacción.

Es decir: **la premisa central del proyecto — regulación específica de tipo celular — nunca
llegó a implementarse**, y con esta arquitectura no podía. Es, probablemente, el hallazgo más
generalizable de toda la auditoría.

---

## Por qué probablemente no se puede arreglar con más modelo

Las interacciones miRNA→gen provienen de **miRDB v6.0**, que son **predicciones basadas en
complementariedad de secuencia** (no interacciones validadas experimentalmente — corregimos
esa afirmación, que estaba mal en el reporte anterior).

**Pero nuestro grafo no contiene ninguna información de secuencia.** Las características del
gen son 1-D (expresión media) y las del miRNA son un vector aprendido sin contenido
biológico. Para un par que el modelo nunca ha visto, **no hay señal de la que pueda
generalizar** salvo la topología de la red.

Por eso ~0.63 parece ser el **techo de información de la tarea tal como está planteada**,
y no un defecto de la arquitectura. Cambiar el modelo no lo va a mover.

---

## Qué significa esto para el proyecto

### Se suspende (no se descarta: se suspende)

Todos los circuitos regulatorios concretos — **hsa-miR-23a-3p → CCL7** (score 0.986),
**hsa-miR-146a-3p** en Th17, **hsa-miR-140-5p** en oligodendrocitos, y toda la tabla
`top_circuits_by_celltype.tsv` — están **ordenados por el mismo cabezal de predicción** cuya
generalización real es apenas superior a una heurística trivial.

**No son hallazgos validados y no deben presentarse como tales.** Hemos **suspendido la
validación externa contra literatura**: contrastarlos ahora arriesgaría validar un artefacto,
y eso gasta la credibilidad de los colaboradores, no solo horas de GPU.

### Se mantiene

- **La clasificación de tipos celulares (0.9916)**, evaluada con una partición correcta a
  nivel de célula. Es sólida.
- **Toda la infraestructura de auditoría**, que es ahora el activo principal del proyecto: la
  partición correcta de aristas, la prueba automática de fuga, y los tres controles
  independientes.

### El artículo que sí podemos escribir

**El error es el resultado.** Podemos demostrar, de forma limpia, cuantificada y reproducible,
que:

> *la evaluación transductiva infla la predicción de blancos de miRNA de 0.63 a 0.98, y los
> controles no pareados la inflan aún más — hasta el punto de que una heurística de una línea
> alcanza 0.87.*

Este defecto es **muy común** en la literatura de redes neuronales sobre grafos en biología.
Tenemos la demostración completa sobre un grafo biomédico real, más el protocolo corregido
como contribución reutilizable.

**No es el artículo que planeábamos, pero es defendible y nadie lo puede desmontar.** Eso vale
más que un titular que se derrumba en revisión.

---

## Preguntas que probablemente tengan

**¿Todo el trabajo fue en vano?**
No. La clasificación celular es real, el pipeline es reproducible, y la parte de auditoría es
publicable por sí misma. Lo que se pierde es la afirmación de "descubrimiento de circuitos".

**¿No se puede simplemente entrenar más o con un modelo mejor?**
No es un problema de capacidad del modelo: es un problema de **información disponible**. El
grafo no contiene la señal (secuencia) que determina esas interacciones. La vía real sería
**añadir características de secuencia** (semilla del miRNA, 3'UTR del gen), lo que es
esencialmente un proyecto distinto.

**¿Cómo sabemos que 0.6271 es el número correcto y no otro artefacto?**
Porque está verificado por una prueba automática que comprueba que ninguna arista apartada
aparece en la entrada del modelo, en ninguna de las dos direcciones — y porque tres controles
independientes (fuga de mensajes, sesgo de popularidad, heurísticas topológicas) coinciden en
la misma historia.
