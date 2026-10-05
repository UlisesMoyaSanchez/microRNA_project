# Acceso a los datos — auditoría independiente

Complemento de `INDEPENDENT_AUDIT_GUIDE.md` §3. Aquí solo hay **cómo obtener y verificar los datos**; no hay resultados nuestros.

Todo se obtiene de fuentes públicas, fijadas por commit, salvo el **grafo propio**, que los mantenedores entregan por separado (§7).

**Requisitos:** `curl`, Python 3 con numpy; `unzip` o `zipfile` (solo para `cksnp_gnn`); el paquete `ogb` y PyTorch/PyG (solo para los benchmarks OGB).

```bash
mkdir -p data/raw/hmdd_survey && cd data/raw/hmdd_survey
GH=https://raw.githubusercontent.com
```

> Los hashes de §1–§5 salen de los manifiestos que generaron nuestros scripts de descarga (2026-08-13). Si un hash **no** coincide, detengan y avisen (guía §5); no lo "arreglen".

---

## 1. `canonical5430` (MGCNSS, NIMGSA, HLGNN-MDA: una sola matriz)

Forma esperada **495 × 383, 5,430 positivos** (filas = miRNA, columnas = enfermedad). Se bajan los tres archivos, uno por paper, para comprobar que son la misma matriz.

```bash
mkdir -p mgcnss nimgsa hlgnn_mda
curl -sS --fail "$GH/15136943622/MGCNSS/f8a87e698b78696fb8cfe930a2b86ae53e61c81a/data/miRNA_disease_matrix.csv" -o mgcnss/matrix.csv
curl -sS --fail "$GH/zhanglabNKU/NIMGSA/5bf10a93d32286a84bed641d0f85be19f2ba011f/m-d.txt" -o nimgsa/matrix.csv
curl -sS --fail "$GH/LiangYu-Xidian/HLGNN-MDA/e3ac824017b8d1bf246e7f695551a8874d231724/Python/data/5430dataset/1.miRNA-disease%20associations/association.txt" -o hlgnn_mda/matrix.csv
```

SHA-256 de los **bytes crudos tal como los sirve GitHub** (verificado de nuevo el 2026-10-05). El formato difiere entre papers: MGCNSS y NIMGSA van separados por comas, y HLGNN-MDA va separado por **espacios** y en notación científica (`1.000000000000000000e+00`):

| Archivo | SHA-256 |
|---|---|
| `mgcnss/matrix.csv` | `03e872210f54e1bc1630328a45ec16e6846f5065f6da6f80aabfc5c52665a2b2` |
| `nimgsa/matrix.csv` | `c635d9116fc1fc6c212f22410af5d11434a46b8ad128cb049d236e22216bee06` |
| `hlgnn_mda/matrix.csv` | `095a11be02b515ab3e392b72dccc40acc1731144750d4578082f56fa04f1b112` |

**Verificación de identidad (primer paso pedido en la guía).** Los tres deben dar el mismo hash de valores parseados, `78f1a3cd7a8b62cf6341208f2c651f732e7a28f391d93125d46f8fb3aa77d04e`, calculado así:

```python
import json, hashlib
import re
def parsed_hash(path):
    rows = [[int(float(v)) for v in re.split(r"[,\s]+", ln.strip())]
            for ln in open(path).read().splitlines() if ln.strip()]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()
```

Si prefieren otra definición de "valores parseados" (por ejemplo `np.array_equal` entre las tres), basta con que las tres matrices sean idénticas y den 495 × 383 con 5,430 unos.

*(Archivos opcionales, no necesarios para la pista A: `train7.txt` y `test7_1.txt` en `MGCNSS/.../data/train/`, y `train5430_idx.txt` y `test2792_idx.txt` en `HLGNN-MDA/Python/data/`. Son los splits que publicaron esos papers; verlos antes de la Fase 3 puede anclar su implementación.)*

## 2. `digamn` (DiGAMN)

**917 × 792, 14,550 positivos.** Matriz densa 0/1 separada por comas, sin encabezado.

```bash
mkdir -p digamn
curl -sS --fail "$GH/yinboliu-git/GAMN/6695f4e4b4f3ff7060d934a43807ae25925c574e/data/MDA-CF/m_d.csv" -o digamn/matrix.csv
```

SHA-256 de los bytes: `354f0aa1e77d1da348405b7f23421e6d24e678b9c6fdbadebfc0bcba37667688`.

## 3. `cksnp_gnn` (CKSNP-GNN)

**901 × 877, 16,427 positivos.**

**El zip pesa ~112 MB y el servidor es lento**; si la descarga se corta, reanúdenla con `curl -C -`.

```bash
mkdir -p cksnp_gnn
curl -sS --fail http://public.aibiochem.net/DNA_RNA/Genes_Human-miRNA-disease-Associations/code.zip -o cksnp_gnn/code.zip
sha256sum cksnp_gnn/code.zip
unzip -p cksnp_gnn/code.zip data/all_mirna_disease_pairs.csv > cksnp_gnn/pairs.csv
```

SHA-256 del zip: `3277b6661302dca84d745819ef61b281b285a55a45678e826269039956122952`.

**Formato de `pairs.csv`:** una arista por línea, `miRNA,enfermedad,etiqueta`, **índices desde 1**, sin encabezado. Los positivos son las filas con etiqueta `1` (deben ser 16,427); la forma de la matriz es el máximo de cada columna (901 y 877). **Ojo:** el README y el código del zip describen otro modelo (GAEMDA); solo se usa la matriz. El servidor es HTTP, sin TLS, y es un espejo institucional: si no responde, avisen.

## 4. `meahne` (MEAHNE)

**1,296 miRNAs × 11,783 enfermedades, 17,972 positivos.**

```bash
mkdir -p meahne
for f in "DATA/node_list/mir.csv" "DATA/node_list/disease.csv" "DATA/association/mir_disease%20.csv"; do
  curl -sS --fail "$GH/yyx-hc/MEAHNE/96f3faa6e143b6e70ef29c9692ffa399c2b5780a/$f" -o "meahne/$(basename "${f//%20/}")"
done
```

(El nombre upstream del tercer archivo lleva **un espacio**: `mir_disease .csv`; arriba se guarda sin él.)

**Formato:**
- `mir.csv` y `disease.csv` tienen un encabezado; el número de nodos es el número de líneas menos 1.
- `mir_disease.csv` tiene encabezado `,mir,disease` y filas `fila,id_mirna,id_enfermedad` con ids **desde 0**; hay que descartar la primera columna.

No hay hash de bytes crudos para estos tres archivos en nuestros manifiestos; la verificación es la forma (1,296 × 11,783) y los 17,972 positivos.

## 5. `couplemda` (CoupleMDA)

**2,090 miRNAs × 1,754 enfermedades; 13,509 positivos de entrenamiento y 1,523 de prueba**, tras quitar 43 pares que aparecen en ambos archivos.

```bash
mkdir -p couplemda
B=$GH/lizhj39/CoupleMDA/c9128cb789cf32c13ad5e381d2055a03f91dcbe8/data/Zou
for f in node.dat link.dat link.dat.test info.dat; do curl -sS --fail "$B/$f" -o couplemda/$f; done
```

**Formato (formato HGTMDA):** archivos con CRLF; columnas separadas por tabulador.
- `link.dat` / `link.dat.test`: `origen \t destino \t tipo_de_enlace \t peso`. Se usan solo las filas con **tipo `12`** (miRNA–enfermedad). Los ids son **globales**: miRNA = 0–2089 (id local = id global), enfermedad = 2090–3843 (id local = id global − 2090).
- `link.dat.test` solo tiene positivos (1,529 filas, 1,523 pares únicos); `link.dat`, 13,744 filas de tipo 12 (13,552 pares únicos). Hay que **deduplicar** pares antes de contar.
- Los 43 pares presentes en ambos archivos se **excluyen del entrenamiento**; verifiquen que obtienen 13,509 y 1,523.
- Los negativos de CoupleMDA no están publicados (los genera su cargador en tiempo de ejecución); cada auditor genera los suyos.

## 6. Benchmarks OGB (`ogbl-ddi`, `ogbl-ppa`)

```bash
pip install ogb
python - <<'EOF'
from ogb.linkproppred import PygLinkPropPredDataset
for name in ("ogbl-ddi", "ogbl-ppa"):
    ds = PygLinkPropPredDataset(name=name, root="data/raw/ogb")
    print(name, ds[0].num_nodes, ds.get_edge_split().keys())
EOF
```

Se usa el split oficial. `ogbl-ddi` tiene 4,267 nodos. **Memoria:** ogbl-ppa necesitó ~31 GB de RAM en nuestro baseline de heurísticas; si no la tienen, audítenlo solo en ddi y declárenlo. Se necesitan ~20 GB de disco libres para ambos.

---

## 7. El grafo propio (lo entregan los mantenedores)

Cohorte de scRNA-seq + miRDB v6.0; **~550 MB, no versionado**; 2,460 miRNAs × 3,000 genes, 44,186 aristas. **No** se descarga de ningún lado: lo envía el equipo con su SHA-256, que calcula al congelar.

Campos que completan los mantenedores antes de enviar la guía:

- Medio de entrega: `________` (enlace, Zenodo, rsync…)
- Nombre del archivo y formato: `________`
- SHA-256: `________`
- Commit de congelamiento del repositorio: `________`

Quien no lo reciba puede auditar C1 solo sobre los cinco grafos externos y debe declararlo (guía §3).

---

## 8. Resumen de verificación

| Grafo | Forma | Positivos | Verificación |
|---|---|---|---|
| `canonical5430` | 495 × 383 | 5,430 | las 3 matrices idénticas; hash parseado `78f1a3cd…77d04e` |
| `digamn` | 917 × 792 | 14,550 | SHA-256 `354f0aa1…667688` |
| `cksnp_gnn` | 901 × 877 | 16,427 | SHA-256 del zip `3277b666…122952` |
| `meahne` | 1,296 × 11,783 | 17,972 | forma y conteo |
| `couplemda` | 2,090 × 1,754 | 13,509 train + 1,523 test | forma, conteos y 43 solapados |
| `ogbl-ddi` | 4,267 nodos | — | split oficial |
| grafo propio | 2,460 × 3,000 | 44,186 aristas | SHA-256 de los mantenedores |

**Si algo no coincide, detengan y avisen** (guía §5). No ajusten los datos para que coincidan.
