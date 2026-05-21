# Goalkeeper 1v1 Shot Analysis Pipeline

Pipeline en Python para transformar videos de jugadas 1v1 de futbol en datos estructurados. Esta primera version implementa solamente la Etapa 1:

```text
video -> tracking del balon -> deteccion del tiro -> fin de jugada -> area penal/porteria -> CSVs
```

No incluye deteccion del portero, pose estimation, poligonos corporales ni probabilidad de alcance. Esas partes quedan preparadas como extensiones futuras.

## Uso rapido

Instalacion sugerida:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_setup.py
```

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

Con una URL individual:

```bash
python src/shot_detector.py \
  --video-url "https://example.com/video.mp4" \
  --model models/ball_detector/ball_model.pt \
  --config configs/pitch_config_example.json \
  --frame-output outputs/frame_level/url_video_ball_tracking.csv \
  --shot-output outputs/shot_level/url_video_shot_tuple.csv \
  --url-cache-dir data/raw_videos/url_cache
```

Si YouTube pide login o aparece `HTTP Error 429`, corre con cookies del
navegador donde ya estes logueado y descarga por tandas:

```bash
python scripts/extract_frames.py \
  --no-auto-url-lists \
  --url-list data/raw_videos/highlights.txt \
  --url-start-index 1 \
  --max-downloads 5 \
  --cookies-from-browser chrome \
  --sleep-requests 2 \
  --sleep-interval 20 \
  --max-sleep-interval 60 \
  --fps 3 \
  --max-frames 250 \
  --sampling-mode uniform
```

Puedes cambiar `chrome` por `safari` o `firefox`. Evita `--force-download`
si ya hay videos en cache.

Sin video anotado:

```bash
python src/shot_detector.py \
  --video data/raw_videos/input.mp4 \
  --model models/ball_detector/ball_model.pt \
  --config configs/pitch_config_example.json \
  --frame-output outputs/frame_level/ball_tracking.csv \
  --shot-output outputs/shot_level/shot_tuple.csv
```

Tambien se puede ejecutar con el wrapper:

```bash
python scripts/run_shot_detector.py --video data/raw_videos/input.mp4 --model models/ball_detector/ball_model.pt --config configs/pitch_config_example.json --frame-output outputs/frame_level/ball_tracking.csv --shot-output outputs/shot_level/shot_tuple.csv
```

Procesar una carpeta completa:

```bash
python scripts/batch_process_videos.py \
  --input-dir data/raw_videos \
  --model models/ball_detector/ball_model.pt \
  --config configs/pitch_config_example.json \
  --output-dir outputs
```

Si usas `--save-review-frames`, se guardan ventanas alrededor del tiro y del fin estimado de la jugada.

Procesar URLs desde un `.txt`:

1. Crea un archivo en `data/raw_videos/`, por ejemplo:

   ```text
   data/raw_videos/video_urls.txt
   ```

2. Pon una URL por linea:

   ```text
   https://example.com/video_001.mp4
   https://example.com/video_002.mp4
   https://example.com/video_003.mp4
   ```

3. Corre el batch. El script detecta automaticamente los `.txt` dentro de `data/raw_videos/`, descarga cada URL a `data/raw_videos/url_cache/` y luego procesa los videos descargados:

   ```bash
   python scripts/batch_process_videos.py \
     --input-dir data/raw_videos \
     --model models/ball_detector/ball_model.pt \
     --config configs/pitch_config_example.json \
     --output-dir outputs \
     --annotated \
     --review-frames
   ```

Tambien puedes pasar una lista explicita:

```bash
python scripts/batch_process_videos.py \
  --url-list data/raw_videos/video_urls.txt \
  --model models/ball_detector/ball_model.pt \
  --config configs/pitch_config_example.json \
  --output-dir outputs
```

## Configuracion

El archivo `configs/pitch_config_example.json` define:

- `video_id`
- `penalty_area_polygon`
- `goal_corners`

La porteria se normaliza asi:

```text
bottom_left  -> (0, 0)
bottom_right -> (1, 0)
top_right    -> (1, 1)
top_left     -> (0, 1)
```

Las coordenadas principales de salida son continuas:

```text
goal_entry_u = 0.0 izquierda, 0.5 centro, 1.0 derecha
goal_entry_v = 0.0 suelo, 0.5 media altura, 1.0 travesano
```

Las zonas discretas (`left`, `center`, `right`, `low`, `middle`, `high`) se derivan de esas coordenadas y no reemplazan los valores continuos.

## Salidas

### Frame-level

Una fila por frame con tracking, suavizado, velocidad, aceleracion, flags de candidato, tiro seleccionado y fin de jugada.

Nota importante: el tracking y la cinemática se calculan frame por frame porque hacen falta para detectar el tiro. La geometría del polígono del área penal no se evalúa en todos los frames; `inside_penalty_area` solo se calcula en eventos relevantes:

- frame del tiro seleccionado;
- frame estimado de conclusión de la jugada.

En el resto de frames queda `NA`.

### Shot-level

Una fila por video/tiro con el frame detectado, posicion y velocidad del balon, area penal al momento del tiro, fin estimado de jugada, outcome (`goal`, `stopped_or_save`, `out_or_miss`, `unknown`), punto de entrada, coordenadas normalizadas, metros aproximados, zonas derivadas, confianza y notas.

## Estructura

```text
goalkeeper-1v1-analysis/
├── configs/
├── data/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
└── tests/
```

## Tests

```bash
python -m pytest
```

## Entrenar El Detector YOLO Del Balon

El pipeline principal espera el modelo final en:

```text
models/ball_detector/ball_model.pt
```

Para crearlo, usa este flujo.

### 1. Extraer Frames

```bash
python scripts/extract_frames.py --fps 2 --max-frames 400
```

Lee videos locales de `data/raw_videos/`. Si hay archivos `.txt` con URLs en esa carpeta, tambien descarga esas URLs a `data/raw_videos/url_cache/` antes de extraer frames.

Por defecto, los frames se muestrean de forma uniforme a lo largo del video.
Esto es importante para evitar quedarnos solo con el inicio de partidos largos.
Para rehacer la extraccion limpiando frames anteriores:

```bash
python scripts/extract_frames.py \
  --fps 2 \
  --max-frames 400 \
  --sampling-mode uniform \
  --clear-output
```

Para highlights o clips cortos, pon sus URLs en `data/raw_videos/highlights.txt`
y corre:

```bash
python scripts/extract_frames.py \
  --no-auto-url-lists \
  --url-list data/raw_videos/highlights.txt \
  --url-start-index 1 \
  --max-downloads 5 \
  --cookies-from-browser chrome \
  --sleep-requests 2 \
  --sleep-interval 20 \
  --max-sleep-interval 60 \
  --fps 3 \
  --max-frames 250 \
  --sampling-mode uniform \
  --clear-output
```

Notas para YouTube:

- Para extraer frames no se necesita audio; el descargador usa video-only por defecto.
- Si ya aparecio `HTTP Error 429` o `Sign in to confirm you're not a bot`, espera 30-60 minutos antes de reintentar.
- Trabaja en tandas de 5-10 URLs con `--max-downloads`; no conviene bajar 50-100 seguidas.
- Para retomar otra tanda, cambia `--url-start-index`; por ejemplo, usa `--url-start-index 31 --max-downloads 5` para intentar URLs 31-35.
- Si las cookies del navegador fallan, exporta un `cookies.txt` en formato Netscape y usa `--cookies-file ruta/al/cookies.txt`.

### 2. Seleccionar Frames Utiles

```bash
python scripts/select_frames.py --blur-thresh 80 --hash-thresh 8 --max-total 800
```

Si regeneraste frames, limpia tambien la seleccion anterior:

```bash
python scripts/select_frames.py \
  --blur-thresh 80 \
  --hash-thresh 8 \
  --max-total 1000 \
  --clear-output
```

La salida queda en:

```text
data/frames_to_annotate/
```

### 3. Anotar

Anota externamente en CVAT, Label Studio o Roboflow.

Reglas:

- usar una sola clase: `ball`;
- caja ajustada al balon visible;
- no anotar cabezas, zapatos, guantes, lineas ni manchas blancas;
- incluir balones pequenos, borrosos, parcialmente ocluidos y cerca del poste.

### 4. Convertir Anotaciones

CVAT:

```bash
python scripts/convert_annotations.py \
  --format cvat_xml \
  --annotations data/annotations/cvat_export.xml \
  --images data/frames_to_annotate/
```

Label Studio:

```bash
python scripts/convert_annotations.py \
  --format labelstudio \
  --annotations data/annotations/labelstudio_export.json \
  --images data/frames_to_annotate/
```

Roboflow YOLO:

```bash
python scripts/convert_annotations.py \
  --format roboflow_yolo \
  --roboflow-dir data/annotations/roboflow_export/
```

Esto crea el dataset YOLO en `data/ball_dataset/`.

### 5. Diagnosticar Labels

```bash
python tools/check_labels.py
python tools/preview_dataset.py --split train --n 16
```

### 6. Entrenar

```bash
python scripts/train_ball_detector.py \
  --data data/ball_dataset/data.yaml \
  --base-model yolov8n.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --output models/ball_detector/ball_model.pt
```

El script valida que el dataset no este vacio, entrena YOLO, busca `best.pt`, lo copia a `models/ball_detector/ball_model.pt` y ejecuta validacion final.

### 7. Validar e Inferir

```bash
python scripts/validate_model.py --split test --preview 20
python tools/run_inference.py --video data/raw_videos/jugada.mp4
```

La validacion guarda metricas globales del modelo en:

```text
outputs/model_reports/ball_detector_metrics.json
outputs/model_reports/ball_detector_metrics.csv
```

Estas metricas globales incluyen precision, recall, mAP50 y mAP50-95. Las confianzas por observacion quedan dentro de las bases, por ejemplo `ball_confidence`, `shot_ball_confidence`, `shot_confidence` y `goal_entry_confidence`.

Con una URL:

```bash
python tools/run_inference.py --video "https://example.com/video.mp4"
```

## Demo Rapido Del Poligono Del Portero

Para una presentacion, se puede mostrar un poligono/circulo proxy del portero
sin tener todavia el modelo de pose. Si tienes un tracking estilo SoccerNet/MOT
con bbox del portero:

```bash
python scripts/demo_keeper_polygon.py \
  --video data/raw_videos/jugada.mp4 \
  --tracking-file data/annotations/soccernet_tracking.txt \
  --config configs/pitch_config_example.json \
  --frame 143 \
  --out-image outputs/review_frames/keeper_polygon_demo.jpg \
  --out-csv outputs/keeper_polygon_demo.csv
```

Si no tienes tracking, puedes pasar una bbox manual `x y w h`:

```bash
python scripts/demo_keeper_polygon.py \
  --video data/raw_videos/jugada.mp4 \
  --keeper-bbox 520 220 80 180 \
  --frame 143
```

Esto genera una imagen anotada y un CSV con cinco puntos proxy, features del
poligono y un circulo/ellipse de alcance. Es demo visual, no estimacion final.
Pasa `--config` solo cuando el archivo este calibrado para ese video.

## Demo Web De Regiones De Probabilidad

Hay un demo interactivo en:

```text
web_demo/keeper_probability_demo.html
```

La forma recomendada de correrlo es con un servidor local desde la raiz del
repo. Esto permite que el navegador cargue correctamente
`web_demo/save_probability_model.json`:

```bash
python -m http.server 8010 --directory web_demo
```

Luego abre:

```text
http://127.0.0.1:8010/keeper_probability_demo.html
```

Si el puerto `8010` ya esta ocupado, usa otro, por ejemplo:

```bash
python -m http.server 8011 --directory web_demo
```

El archivo tambien puede abrirse con `file://`, pero el servidor local es mas
confiable para que el modo `Baseline ML (PFF)` cargue el JSON entrenado.

El demo permite manipular:

- posicion del portero;
- posicion del balon al tiro;
- punto objetivo dentro de la porteria;
- tiempo estimado al arco;
- incertidumbre Monte Carlo.

El canvas muestra el area grande completa, area chica, punto penal, posicion
original del tiro, punto objetivo y heatmap de `P(save)`.

El sitio incluye dos modos:

- `Baseline ML (PFF)`: regresion logistica entrenada con tiros gol/parada de
  PFF World Cup 2022 y exportada a `web_demo/save_probability_model.json`.
- `Simulador geometrico`: funcion proxy no entrenada para explicar alcance,
  deformacion y reaccion.

### Capas Analiticas Del Demo

El panel `Incertidumbre y decision` agrega cuatro capas sobre la probabilidad
base:

```text
Monte Carlo      simula tiros cercanos al objetivo
Alpha / IC       cambia el intervalo central reportado
Entropia         mide incertidumbre binaria parada/no parada
Optimizacion     busca una posicion valida del portero
```

La optimizacion no evalua infinitos puntos continuos. Aproxima el optimo con
una malla discreta y despues refina localmente alrededor del mejor punto. En la
UI se reporta cuantas posiciones fueron evaluadas y cuantos samples Monte Carlo
se usaron por posicion.

El panel `Resultados interpretables del modelo` muestra metricas del modelo,
una lectura narrativa y las variables que mas empujan el logit hacia parada o
hacia no parada.

### Reentrenar El Modelo Del Demo

Para probar la tuberia de ML completa sin datos reales:

```bash
python scripts/train_save_probability_model.py \
  --demo-synthetic \
  --n-synthetic 2500 \
  --out-training-csv data/processed/save_probability_demo_training.csv
```

Esto entrena una regresion logistica demo y genera:

```text
models/save_probability/save_probability_model.pkl
outputs/model_reports/save_probability_metrics.json
web_demo/save_probability_model.json
```

Para entrenar con datos reales, usa una tabla con columnas tipo:

```text
keeper_center_u
keeper_center_v
keeper_body_width
keeper_body_height
keeper_hand_span
keeper_polygon_area_uv
keeper_pose_confidence
goal_entry_u
goal_entry_v
shot_ball_speed
time_ball_to_goal
reaction_time
outcome
```

Y corre:

```bash
python scripts/train_save_probability_model.py \
  --data data/processed/annotated_1v1_shots.csv \
  --outcome-col outcome
```

Valores positivos de `outcome`: `save`, `stopped_or_save`. Valor negativo
principal: `goal`. Los tiros fuera deben separarse o tratarse como otro modelo.

Despues de entrenar o editar el demo, valida que el HTML y el JSON del modelo
sean consistentes:

```bash
node scripts/_smoke_pitch_demo.mjs
python -m pytest -q
```

### PFF FC World Cup 2022

El camino recomendado para entrenar el modo `Baseline ML (PFF)` es el dataset
PFF FC World Cup 2022, porque combina eventos de tiro con tracking. Como PFF lo
entrega por solicitud/formulario, coloca el dump localmente en:

```text
data/pff_worldcup_2022/
```

El importador acepta un `events.json` global o varios `*.json` por partido,
como el dump PFF FC que trae 64 archivos de la Copa del Mundo.

Despues construye la tabla entrenable:

```bash
python scripts/build_pff_save_probability_dataset.py \
  --pff-root data/pff_worldcup_2022 \
  --tracking-dir data/pff_worldcup_2022/tracking \
  --out data/processed/pff_worldcup_2022_save_probability.csv
```

Y entrena el modelo del demo web con esa tabla:

```bash
python scripts/train_save_probability_model.py \
  --data data/processed/pff_worldcup_2022_save_probability.csv \
  --outcome-col outcome \
  --out-training-csv data/processed/pff_worldcup_2022_training_used.csv
```

El detalle operativo esta en `docs/pff_worldcup_2022_workflow.md`. El importador
marca `keeper_source` y `tracking_used`; si aparece `fallback_center`, esa fila
todavia no tiene posicion real del portero por tracking.
