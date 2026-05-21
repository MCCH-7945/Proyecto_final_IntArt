# PFF FC World Cup 2022 workflow

Este repo no descarga PFF automaticamente porque el dataset se entrega por
solicitud/formulario. Cuando lo tengas, colocalo localmente asi:

```text
data/pff_worldcup_2022/
├── 3812.json
├── 3813.json
├── ...
├── 10517.json
├── PFF FC Event Data Specification v2.4.pdf
```

Si PFF entrega otro formato, tambien se acepta:

```text
data/pff_worldcup_2022/
├── events.json
├── metadata/
├── rosters/
└── tracking/
    ├── <game_id>.jsonl.bz2
    └── ...
```

La pieza obligatoria para empezar es el event data: ya sea un `events.json`
global o varios `*.json` por partido. Si existe `tracking/<game_id>.jsonl.bz2`
o `tracking/<game_id>.jsonl`, el importador busca el `frame_id` del tiro y
extrae portero + balon desde tracking. Con el dump actual, las posiciones vienen
en `homePlayers`, `awayPlayers` y `ball` dentro de cada evento, así que el
importador extrae un proxy event-level del portero.

## 1. Construir tabla entrenable

```bash
python scripts/build_pff_save_probability_dataset.py \
  --pff-root data/pff_worldcup_2022 \
  --tracking-dir data/pff_worldcup_2022/tracking \
  --out data/processed/pff_worldcup_2022_save_probability.csv
```

Salida principal:

```text
data/processed/pff_worldcup_2022_save_probability.csv
```

Columnas clave:

```text
keeper_center_u
keeper_center_v
keeper_pose_confidence
keeper_source
tracking_used
ball_position_u
ball_position_v
goal_entry_u
goal_entry_v
shot_ball_speed
time_ball_to_goal
outcome
save_label
```

`save_label = 1` significa parada del portero. `save_label = 0` significa gol.
Tiros fuera, bloqueados por defensores o sin etiqueta clara se excluyen por
defecto para que el primer modelo sea `save vs goal`.

Para conservar tambien tiros no etiquetables:

```bash
python scripts/build_pff_save_probability_dataset.py \
  --pff-root data/pff_worldcup_2022 \
  --include-unlabeled \
  --out data/processed/pff_worldcup_2022_shots_all.csv
```

## 2. Entrenar el modelo real

```bash
python scripts/train_save_probability_model.py \
  --data data/processed/pff_worldcup_2022_save_probability.csv \
  --outcome-col outcome \
  --out-training-csv data/processed/pff_worldcup_2022_training_used.csv
```

Esto actualiza:

```text
models/save_probability/save_probability_model.pkl
outputs/model_reports/save_probability_metrics.json
web_demo/save_probability_model.json
```

Despues abre:

```text
web_demo/keeper_probability_demo.html
```

El demo intentara cargar `web_demo/save_probability_model.json`. Si ese archivo
fue generado con PFF, el modo `ML demo` ya usara coeficientes entrenados con los
datos reales disponibles.

## Limitaciones importantes

- Si `keeper_source = fallback_center`, esa fila no contiene tracking real del
  portero.
- Si `shot_speed_source = default_24_mps`, la velocidad fue aproximada.
- Si `time_to_goal_source = distance_over_speed`, el tiempo al arco fue
  derivado de distancia y velocidad, no observado directamente.
- Si el tracking real esta en un layout distinto, pasa `--events` y
  `--tracking-dir` explicitamente. El importer es tolerante, pero puede requerir
  un ajuste menor si PFF entrega un schema diferente.
