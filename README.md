# Keeper Save Probability Demo

Demo interactivo para estimar y explicar la probabilidad de que un portero pare
un tiro en una situacion 1v1. El foco principal del repo, para presentacion, es
el demo web:

```text
web_demo/keeper_probability_demo.html
```

El demo combina:

- un modelo LOGIT entrenado sobre tiros gol/parada;
- una visualizacion espacial de `P(save)`;
- Monte Carlo para incertidumbre del tiro;
- entropia en bits para incertidumbre de la decision;
- optimizacion de posicion del portero;
- explicabilidad por contribuciones del modelo.

La version actual es una linea base interpretable. No debe presentarse como un
modelo biomecanico final del portero.

## Correr El Demo

Desde la raiz del repo:

```bash
cd "/Users/malikcorverachoi/Documents/New project/goalkeeper-1v1-analysis"
python -m http.server 8010 --directory web_demo
```

Luego abre:

```text
http://127.0.0.1:8010/keeper_probability_demo.html
```

Si el puerto `8010` esta ocupado:

```bash
python -m http.server 8011 --directory web_demo
```

y abre:

```text
http://127.0.0.1:8011/keeper_probability_demo.html
```

El servidor local es preferible a abrir el HTML con `file://`, porque permite
cargar correctamente:

```text
web_demo/save_probability_model.json
```

Ese JSON contiene los coeficientes, escalamiento, metricas y metadatos del
modelo que usa el demo.

## Instalacion Para Validar O Reentrenar

Para solo abrir la pagina web, basta con el servidor anterior. Para correr
tests, procesar datos o reentrenar modelos:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_setup.py
```

Validacion rapida del demo:

```bash
node scripts/_smoke_pitch_demo.mjs
python -m pytest -q
```

Resultado esperado en la version actual:

```text
27 passed
```

## Que Muestra La Pagina

La pagina dibuja una vista normalizada del area frente a la porteria:

- area grande;
- area chica;
- punto penal;
- posicion del balon al momento del tiro;
- posicion del portero;
- punto objetivo del tiro dentro de la porteria;
- heatmap de probabilidad de parada.

El punto amarillo representa la posicion original del tiro. El punto blanco
dentro del arco representa el objetivo o punto estimado de entrada del balon.

Las tarjetas superiores muestran:

```text
P parada objetivo   probabilidad estimada en el punto blanco
P >= 0.80           porcentaje de la porteria con alta probabilidad de parada
P >= 0.50           porcentaje de la porteria con probabilidad media/alta
```

## Modelo Principal: LOGIT

El modo principal del demo es:

```text
Baseline ML (PFF)
```

Usa una regresion logistica para estimar:

```text
Pr(parada)
```

La variable objetivo se codifica como:

```text
save_label = 1  tiro parado
save_label = 0  gol
```

Conceptualmente, el modelo calcula:

```text
logit = beta_0 + beta_1 x_1 + beta_2 x_2 + ... + beta_k x_k
```

y transforma ese valor a probabilidad:

```text
Pr(parada) = 1 / (1 + exp(-logit))
```

Variables usadas por el modelo/demostracion:

```text
keeper_center_u
keeper_center_v
ball_position_u
ball_position_v
goal_entry_u
goal_entry_v
distance_keeper_to_ball_m
distance_keeper_to_target_m
distance_ball_to_target_m
lateral_delta_m
vertical_delta_m
time_ball_to_goal
shot_ball_speed
keeper_outside_goal
```

Las variables `u` y `v` son coordenadas normalizadas respecto a la porteria:

```text
goal_entry_u = 0.0 poste izquierdo, 0.5 centro, 1.0 poste derecho
goal_entry_v = 0.0 suelo, 1.0 travesano
```

### Metricas Del Modelo Actual

El modelo exportado en `web_demo/save_probability_model.json` fue generado a
partir de la tabla PFF procesada. En la version actual:

```text
n ~= 577 tiros etiquetados
ROC AUC ~= 0.545
Brier ~= 0.265
```

Interpretacion honesta:

```text
El modelo aprende una senal debil/modesta. Sirve como baseline explicable para
el demo, pero no como modelo final de decision del portero.
```

## Capas Analiticas Del Demo

### 1. Probabilidad Base

Es la salida directa del LOGIT:

```text
P(save | estado del tiro y portero)
```

Esta probabilidad se recalcula cuando cambian la posicion del portero, el balon,
el objetivo o el tiempo al arco.

### 2. Monte Carlo

El demo no asume que el tiro llega exactamente al punto blanco. Genera muchos
tiros cercanos alrededor del objetivo:

```text
(u, v), (u + ruido, v + ruido), ...
```

Para cada tiro simulado calcula `P(save)` y resume:

```text
media Monte Carlo
intervalo inferior
intervalo superior
ancho del intervalo
```

La barra `Incertidumbre` controla la dispersion de esa nube. No infla la
probabilidad base de forma artificial.

### 3. Alpha E Intervalos

Los botones de alpha controlan el intervalo central de la simulacion:

```text
alpha = 0.10  IC 90%
alpha = 0.05  IC 95%
alpha = 0.01  IC 99%
```

Menor alpha implica un intervalo mas amplio. Alpha no cambia el modelo ni la
probabilidad media.

### 4. Entropia

La entropia mide la incertidumbre de la decision binaria:

```text
parada vs no parada
```

Formula:

```text
H(p) = -p log2(p) - (1-p) log2(1-p)
```

Si `p = 0.5`, la incertidumbre es maxima:

```text
H = 1 bit
```

Si `p` esta cerca de `0` o `1`, la entropia baja. Esto significa que el modelo
esta mas decidido, aunque no garantiza que este correcto.

### 5. Optimizacion

La optimizacion busca una posicion valida del portero que maximice la
probabilidad esperada de parada:

```text
max posicion_portero E[P(save)]
```

No evalua infinitos puntos continuos. Usa una aproximacion practica:

```text
malla discreta amplia
-> mejor punto encontrado
-> refinamiento local alrededor del mejor punto
```

La UI muestra cuantas posiciones fueron evaluadas y cuantos samples Monte Carlo
se usaron por posicion.

### 6. Interpretabilidad

El panel `Resultados interpretables del modelo` muestra:

- fuente del modelo;
- AUC;
- Brier score;
- numero de tiros;
- lectura narrativa;
- principales contribuciones al logit.

Para cada variable, la contribucion se calcula como:

```text
coeficiente * valor_escalado
```

Contribucion positiva: empuja hacia mayor `Pr(parada)`.

Contribucion negativa: empuja hacia menor `Pr(parada)`.

Esto explica el comportamiento del modelo, pero no prueba causalidad.

## Reproducir El Modelo PFF

El demo ya incluye un modelo ligero en:

```text
web_demo/save_probability_model.json
```

Por eso la pagina puede correr sin tener el dataset privado PFF en la maquina.

Para reentrenar con el dataset PFF World Cup 2022, coloca el dump local en:

```text
data/pff_worldcup_2022/
```

Luego construye la tabla entrenable:

```bash
python scripts/build_pff_save_probability_dataset.py \
  --pff-root data/pff_worldcup_2022 \
  --tracking-dir data/pff_worldcup_2022/tracking \
  --out data/processed/pff_worldcup_2022_save_probability.csv
```

Y entrena/exporta el modelo:

```bash
python scripts/train_save_probability_model.py \
  --data data/processed/pff_worldcup_2022_save_probability.csv \
  --outcome-col outcome \
  --out-training-csv data/processed/pff_worldcup_2022_training_used.csv
```

Esto genera:

```text
models/save_probability/save_probability_model.pkl
outputs/model_reports/save_probability_metrics.json
web_demo/save_probability_model.json
```

El detalle operativo esta en:

```text
docs/pff_worldcup_2022_workflow.md
```

## Entrenar Un Modelo Demo Sintetico

Si no tienes el dataset PFF pero quieres probar el pipeline de entrenamiento:

```bash
python scripts/train_save_probability_model.py \
  --demo-synthetic \
  --n-synthetic 2500 \
  --out-training-csv data/processed/save_probability_demo_training.csv
```

Advertencia: este modelo sirve solo para pruebas tecnicas y presentacion de
flujo, no para conclusiones empiricas.

## Flujo Sugerido Para Presentar

1. Abrir el demo web.
2. Explicar que el objetivo es estimar `Pr(parada)`.
3. Mover el punto blanco dentro de la porteria.
4. Mostrar como cambia la probabilidad y el heatmap.
5. Activar la lectura de Monte Carlo: media e intervalo.
6. Explicar entropia como incertidumbre parada/no parada.
7. Usar `Aplicar optimo` para mostrar optimizacion de posicion.
8. Leer los factores interpretables del LOGIT.
9. Cerrar con las limitaciones: pocos datos, event-level tracking y falta de pose corporal.

## Limitaciones

- El modelo PFF actual es una linea base, no un predictor definitivo.
- El dataset usado para el demo no contiene pose corporal completa del portero.
- La region visual de alcance no debe interpretarse como poligono corporal real
  entrenado.
- El AUC actual es modesto, por lo que el valor principal del demo es explicar
  el proceso probabilistico e identificar que datos faltan.
- Las recomendaciones de posicion son optimos dentro del modelo, no verdades
  tacticas universales.

## Archivos Principales Del Demo

```text
web_demo/keeper_probability_demo.html    UI interactiva
web_demo/save_probability_model.json     modelo exportado para navegador
web_demo/README.md                       guia especifica del demo
scripts/_smoke_pitch_demo.mjs            validacion rapida del demo
scripts/train_save_probability_model.py  entrenamiento LOGIT
scripts/build_pff_save_probability_dataset.py  conversion PFF -> tabla ML
src/pff_ingestion.py                     lectura flexible de datos PFF
src/reach_probability.py                 dataset/modelo de probabilidad
```

## Pipeline De Video Opcional

El repo tambien conserva scripts para una etapa futura de video:

```text
video -> tracking del balon -> deteccion del tiro -> entrada a porteria
```

Ese pipeline no es necesario para correr el demo web.

Comando base:

```bash
python src/shot_detector.py \
  --video data/raw_videos/input.mp4 \
  --model models/ball_detector/ball_model.pt \
  --config configs/pitch_config_example.json \
  --frame-output outputs/frame_level/ball_tracking.csv \
  --shot-output outputs/shot_level/shot_tuple.csv \
  --annotated-video outputs/annotated_videos/annotated.mp4 \
  --save-review-frames outputs/review_frames/
```

Para crear el detector YOLO del balon:

```bash
python scripts/extract_frames.py --fps 2 --max-frames 400
python scripts/select_frames.py --blur-thresh 80 --hash-thresh 8 --max-total 800
python scripts/convert_annotations.py \
  --format cvat_xml \
  --annotations data/annotations/cvat_export.xml \
  --images data/frames_to_annotate/
python scripts/train_ball_detector.py \
  --data data/ball_dataset/data.yaml \
  --base-model yolov8n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --output models/ball_detector/ball_model.pt
```

## Estructura

```text
goalkeeper-1v1-analysis/
├── web_demo/
├── scripts/
├── src/
├── tests/
├── docs/
├── configs/
├── data/
├── models/
└── outputs/
```
