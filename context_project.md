# Descubrimiento de Redes de Regulación de microARNs en Esclerosis Múltiple mediante Aprendizaje Profundo y Transcriptómica de Célula Única

## 1. Planteamiento de la Problemática
La **Esclerosis Múltiple (EM)** es una enfermedad autoinmune y neurodegenerativa crónica del sistema nervioso central. Su etiología exacta sigue siendo desconocida, lo que dificulta el desarrollo de biomarcadores tempranos y terapias personalizadas eficientes. 

Los **microARNs (miRNAs)** son pequeños ARNs no codificantes que regulan la expresión génica post-transcripcionalmente. Aunque se sabe que alteran las vías inmunitarias, identificar sus dianas terapéuticas exactas presenta tres grandes desafíos computacionales y biológicos:
* **Complejidad de red:** Un solo miRNA puede regular cientos de ARNs mensajeros (mRNAs), y un solo mRNA puede ser regulado por múltiples miRNAs.
* **Falta de especificidad celular:** Los datos masivos tradicionales (*bulk RNA-seq*) promedian la señal de todos los tipos celulares, ocultando cómo los miRNAs regulan específicamente a los linfocitos T, células B o microglía.
* **Limitación de herramientas clásicas:** Los algoritmos basados puramente en secuencias (como TargetScan) generan altas tasas de falsos positivos al no considerar el contexto celular ni la dinámica de unión molecular.

---

## 2. Estado del Arte
La investigación actual se está moviendo rápidamente de los estudios asociativos a los modelos predictivos mecánicos de sistemas:
* **Biomarcadores en Biofluidos:** Estudios recientes han identificado perfiles estables de miRNAs en suero y líquido cefalorraquídeo de pacientes con EM, correlacionados con la progresión de la enfermedad.
* **Modelos de Grafos y Deep Learning:** Se están utilizando Redes Neuronales de Grafos (GNN) y transformadores (como SpliceAI o enfoques similares a ESM) para mapear interacciones ARN-ARN y predecir sitios de unión no canónicos.
* **Transcriptómica de Célula Única (scRNA-seq):** La integración de datos de secuenciación de ARN a nivel de célula única con perfiles de miRNas (*scmiRNA-seq*) está permitiendo reconstruir la heterogeneidad del microambiente inflamatorio en la EM, aunque los datos empíricos de miRNA a nivel de célula única siguen siendo escasos y requieren de imputación computacional avanzada.

---

## 3. Hipótesis
> **Hipótesis de Trabajo:** La integración de perfiles de expresión de microARNs con datos de transcriptómica de célula única (scRNA-seq) mediante arquitecturas de redes neuronales de grafos (GNN) y modelos de lenguaje biológico permite identificar firmas reguladoras de miRNAs específicas de subtipos celulares (ej. células Th17 o microglía reactiva), prediciendo con alta precisión dianas terapéuticas críticas que pasan desapercibidas en análisis transcriptómicos convencionales.

---

## 4. Fuentes de Datos Disponibles (Open Access)
Para entrenar tus modelos de Deep Learning sin un laboratorio físico, puedes explotar repositorios masivos de acceso público:

### Repositorios de Expresión y Transcriptómica
* **NCBI GEO (Gene Expression Omnibus):** Búsqueda de datasets específicos de EM usando palabras clave como `Esclerosis Multiple miRNA`, `scRNA-seq Multiple Sclerosis MS`. (Ej. Datasets de ARN circular/miRNA en células mononucleares de sangre periférica o lesiones cerebrales).
* **EMBL-EBI ArrayExpress:** Repositorio europeo alternativo con datos transcriptómicos crudos y procesados.
* **The Human Cell Atlas / Chan Zuckerberg CellxGene:** Excelente plataforma para descargar matrices masivas de expresión de célula única (scRNA-seq) de pacientes con EM y controles sanos.
* cellxgene_census

### Bases de Datos de Interacción miRNA-Target (Ground Truth)
* **miRTarBase:** La base de datos más grande de interacciones miRNA-target validadas experimentalmente (crucial para el entrenamiento supervisado).
* **DIANA-TarBase / TargetScan / miRDB:** Bases de datos de predicciones computacionales y validaciones biológicas para enriquecer las características (*features*) de tus modelos.
