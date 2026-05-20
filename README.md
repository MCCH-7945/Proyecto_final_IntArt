# Goalkeeper 1v1 Shot Analysis Pipeline

Pipeline en Python para transformar videos de jugadas 1v1 de futbol en datos estructurados. Esta primera version implementa solamente la Etapa 1:

```text
video -> tracking del balon -> deteccion del tiro -> fin de jugada -> area penal/porteria -> CSVs
```

No incluye deteccion del portero, pose estimation, poligonos corporales ni probabilidad de alcance. Esas partes quedan preparadas como extensiones futuras.

## Uso rapido

Instalacion sugerida:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
