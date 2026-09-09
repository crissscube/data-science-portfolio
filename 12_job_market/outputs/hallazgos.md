# Hallazgos — mercado laboral de datos

**Captura:** 2026-09-08 · **Ofertas descargadas:** 1243 · **Ofertas de datos comparables entre regiones:** 271

**Fuentes:** Get on Board (LATAM), Jobicy (remoto global), The Muse (EE. UU.), Arbeitnow (Europa). Todas APIs públicas.

> Generado por `src/analyze.py`. No editar a mano: se regenera.

## Lo que estos datos NO dicen

- Dicen qué **pide** una empresa al publicar. No dicen qué se hace en el puesto.
- Son ofertas **activas** el 2026-09-08 en cuatro portales, no el mercado entero.
- Cada portal tiene su propio sesgo. Por eso las conclusiones fuertes son las que **coinciden en las tres regiones**, no las de una sola.
- La clasificación por tareas es un criterio propio, auditable en `src/extract.py`.

## 0. ¿Piden lo mismo en todo el mundo?

Ofertas por región: LATAM **133** · EE. UU. / Canadá **98** · Europa **40**

**Correlación de Spearman entre el ranking de herramientas de cada región:**

- LATAM ↔ EE. UU./Canadá: **0.80**
- LATAM ↔ Europa: **0.77**
- EE. UU./Canadá ↔ Europa: **0.71**

**El núcleo coincide** — mismas herramientas arriba, en el mismo orden:

| Herramienta | LATAM | EE. UU. / Canadá | Europa |
|---|---|---|---|
| SQL | 63% | 61% | 68% |
| Python | 61% | 72% | 68% |
| AWS | 40% | 19% | 40% |
| Azure | 28% | 19% | 18% |
| GCP | 23% | 16% | 35% |
| Power BI | 30% | 19% | 15% |
| Excel | 14% | 7% | 2% |
| Spark | 23% | 27% | 20% |
| LLM / GenAI | 14% | 28% | 20% |
| PyTorch | 3% | 14% | 5% |
| TensorFlow | 2% | 9% | 0% |
| Deep learning | 4% | 6% | 5% |
| scikit-learn | 4% | 7% | 5% |
| R | 4% | 11% | 0% |
| dbt | 12% | 12% | 25% |
| Snowflake | 9% | 15% | 15% |

**Dónde sí difieren** (lo honesto es nombrarlo):

- Deep learning y PyTorch pesan más en EE. UU. (14%) que en LATAM (3%).
- IA generativa: EE. UU. 28% vs LATAM 14%.
- Power BI y Excel pesan más en LATAM (30% y 14%) que en Europa (15% y 2%).

## 1. El tamaño real de cada puesto

- Ofertas de datos comparables: **271**
- Data Engineer / Arquitecto: **79** (29%)
- Data Analyst / BI: **52** (19%)
- ML / AI Engineer: **49** (18%)
- Business Analyst: **39** (14%)
- Data Scientist: **38** (14%)
- Data Governance / Calidad: **14** (5%)

**Titular:** de 271 ofertas de datos, solo **38** llevan el título de Data Scientist.

## 2. Herramientas (% del total, las tres regiones juntas)

| Herramienta | % |
|---|---|
| Python | 66% |
| SQL | 63% |
| AWS | 32% |
| Power BI | 24% |
| Spark | 24% |
| Azure | 23% |
| GCP | 22% |
| LLM / GenAI | 20% |
| Estadística | 18% |
| Git | 17% |
| Tableau | 14% |
| dbt | 14% |
| Airflow | 14% |
| Snowflake | 12% |
| Looker | 11% |
| Docker | 10% |
| Excel | 10% |
| PyTorch | 7% |
| MLOps | 7% |
| R | 6% |
| scikit-learn | 5% |
| pandas | 5% |
| Deep learning | 5% |
| NoSQL | 4% |
| TensorFlow | 4% |

## 3. Título vs. tareas

De las 38 ofertas tituladas «Data Scientist», clasificadas por tareas:

- Ciencia de datos: **17**
- Mixto: **17**
- Analítica / BI: **4**

## 4. La puerta de entrada

- Ofertas junior: **10** de 271 (3.7%)
- Sin experiencia: **0**
- Mediana de años pedidos por nivel:
  - Experto / Director: 6 años
  - Junior: 1 años
  - Semi Senior: 3 años
  - Senior: 5 años

## 5. Competencia y salario — solo LATAM

> Postulaciones y salario mensual solo existen en Get on Board. Los otros portales publican cifras anuales o ninguna, y mezclarlas daría un número falso.

- Ofertas LATAM: **121**
- Mediana de postulaciones por oferta: **72**
- Máximo observado: **2112**
- Publican salario: **43** (36%)
- Mediana del rango: **3000–3500 USD/mes**

## 6. Habilidades no técnicas

| Habilidad | LATAM | EE. UU. / Canadá | Europa |
|---|---|---|---|
| Comunicar resultados | 62% | 80% | 50% |
| Stakeholders / negocio | 59% | 78% | 60% |
| Trabajo en equipo | 65% | 86% | 62% |
