# Diseno Del Portero: 5 Puntos, Porteria Y Poligono ML

Este documento define la siguiente capa del proyecto: extraer cinco puntos del
portero, normalizarlos respecto a la porteria y construir un poligono corporal
usable para modelos predictivos.

La idea central es que el frame del tiro, ya detectado por el pipeline del
balon, sea el ancla temporal para estudiar al portero.

```text
video
-> tracking del balon
-> shot_frame
-> ventana alrededor del tiro
-> 5 puntos del portero
-> coordenadas relativas a la porteria
-> poligono corporal
-> features ML
```

## 1. Cinco Puntos Corporales

Los puntos minimos son:

```text
head
left_hand
right_hand
left_foot
right_foot
```

En pixeles:

```text
head_x, head_y
left_hand_x, left_hand_y
right_hand_x, right_hand_y
left_foot_x, left_foot_y
right_foot_x, right_foot_y
```

Cada punto debe tener tambien una confianza:

```text
head_confidence
left_hand_confidence
right_hand_confidence
left_foot_confidence
right_foot_confidence
```

Esto importa porque no todos los puntos seran visibles en todos los frames.

## 2. Ventana Temporal

No necesitamos pose en todo el video. La ventana util es alrededor del tiro:

```text
pre_window  = 15 a 30 frames antes del tiro
post_window = hasta play_end_frame o 15 a 30 frames despues
```

Campos recomendados:

```text
video_id
frame
time_sec
shot_frame
frame_relative_to_shot
play_end_frame
is_shot_frame
is_play_end_frame
```

## 3. Porteria Como Sistema De Referencia

La porteria ya esta definida con:

```text
bottom_left
bottom_right
top_left
top_right
```

Usamos la misma homografia del modulo `goal_mapping.py` para mapear puntos de
imagen a coordenadas normalizadas:

```text
u = posicion horizontal respecto a la porteria
v = posicion vertical respecto a la porteria
```

Esto produce:

```text
head_u, head_v
left_hand_u, left_hand_v
...
```

Ventaja: el modelo aprende posiciones comparables entre videos aunque cambie el
zoom, la resolucion o la perspectiva.

Interpretacion:

```text
u = 0.0 poste izquierdo
u = 1.0 poste derecho
v = 0.0 suelo
v = 1.0 travesano
```

Valores fuera de `[0, 1]` son utiles tambien: indican que el portero esta fuera
del marco de la porteria, por ejemplo adelantado o lateralizado.

## 4. Poligono Corporal Observado

El poligono observado se construye con:

```text
left_hand -> head -> right_hand -> right_foot -> left_foot
```

Este poligono no es "todo el cuerpo real". Es una aproximacion compacta de la
postura util para analizar alcance.

Features geometricas:

```text
keeper_polygon_area
keeper_body_width
keeper_body_height
keeper_hand_span
keeper_foot_span
keeper_center_x
keeper_center_y
keeper_center_u
keeper_center_v
keeper_body_aspect_ratio
keeper_pose_valid
```

Tambien se puede construir un vector ML:

```text
[
  head_u, head_v,
  left_hand_u, left_hand_v,
  right_hand_u, right_hand_v,
  left_foot_u, left_foot_v,
  right_foot_u, right_foot_v,
  keeper_polygon_area_uv,
  keeper_hand_span_uv,
  keeper_foot_span_uv,
  goal_entry_u,
  goal_entry_v,
  shot_ball_speed,
  time_ball_to_goal
]
```

## 5. ML Supervisado, Predictivo Y No Supervisado

### A. Supervisado Para Detectar Los 5 Puntos

Si anotamos puntos corporales en imagenes, podemos entrenar un modelo de pose:

```text
imagen -> 5 keypoints del portero
```

Opciones:

```text
YOLO pose custom
RTMPose / MMPose
MediaPipe como pseudo-etiquetador inicial
```

Este es el camino mas directo si queremos puntos semanticamente claros:

```text
head, left_hand, right_hand, left_foot, right_foot
```

### B. Predictivo Para Alcance

Una vez tenemos puntos y resultado del tiro:

```text
postura + trayectoria + entrada del balon -> probabilidad de alcanzar
```

Targets posibles:

```text
reached_ball = 1/0
prob_save
distance_to_goal_entry
best_deformation_cost
```

Modelos:

```text
regresion logistica
random forest / xgboost
red neuronal pequena
calibracion probabilistica
```

### C. No Supervisado Para Agrupar Posturas

El no supervisado es muy util para descubrir tipos de postura:

```text
set position
split stance
diving left
diving right
spread block
one-knee block
```

Pero punto clave:

> El aprendizaje no supervisado por si solo no suele descubrir que un punto es
> "mano izquierda" o "pie derecho" de forma confiable. Para semantica corporal,
> necesitamos labels, un modelo preentrenado, o pseudo-labels.

Uso recomendado:

```text
1. detectar/estimar puntos con modelo supervisado o preentrenado
2. normalizar puntos respecto a la porteria
3. clusterizar vectores de postura
4. estudiar que clusters tienen mayor probabilidad de atajada
```

## 6. Dataset Futuro Del Portero

Tabla frame-level de pose:

```text
video_id
frame
time_sec
shot_frame
frame_relative_to_shot
play_end_frame

keeper_detected
keeper_confidence
keeper_pose_confidence
pose_valid_points_count
pose_missing_points_count

head_x
head_y
head_u
head_v
head_confidence

left_hand_x
left_hand_y
left_hand_u
left_hand_v
left_hand_confidence

right_hand_x
right_hand_y
right_hand_u
right_hand_v
right_hand_confidence

left_foot_x
left_foot_y
left_foot_u
left_foot_v
left_foot_confidence

right_foot_x
right_foot_y
right_foot_u
right_foot_v
right_foot_confidence

keeper_polygon_area
keeper_polygon_area_uv
keeper_body_width
keeper_body_height
keeper_hand_span
keeper_foot_span
keeper_center_x
keeper_center_y
keeper_center_u
keeper_center_v
keeper_pose_valid
keeper_polygon_confidence
```

Tabla shot-level enriquecida:

```text
video_id
shot_frame
goal_entry_u
goal_entry_v
shot_ball_speed
play_outcome

keeper_head_u_at_shot
keeper_left_hand_u_at_shot
keeper_right_hand_u_at_shot
keeper_left_foot_u_at_shot
keeper_right_foot_u_at_shot

keeper_polygon_area_uv_at_shot
keeper_pose_confidence_at_shot
keeper_polygon_confidence_at_shot
pose_valid_points_count_at_shot
distance_keeper_center_to_goal_entry
min_distance_body_point_to_goal_entry
pose_cluster
prob_reach_ball
```

## 7. Estrategia Recomendada

Para hacerlo bien:

```text
1. Usar el pipeline actual para detectar shot_frame y goal_entry_u/v.
2. Extraer ventanas alrededor del tiro.
3. Anotar o pseudo-anotar 5 puntos del portero.
4. Normalizar puntos con la homografia de la porteria.
5. Construir poligono y features geometricas.
6. Clusterizar posturas en coordenadas normalizadas.
7. Entrenar un modelo predictivo de alcance/resultado cuando haya labels suficientes.
```

Esto mantiene separadas tres cosas:

```text
percepcion: detectar puntos
geometria: normalizar y construir poligono
decision/alcance: modelar probabilidad
```

## 8. Metricas Globales Vs Confianza Por Observacion

Las metricas globales del modelo se guardan aparte:

```text
outputs/model_reports/ball_detector_metrics.json
outputs/model_reports/ball_detector_metrics.csv
outputs/model_reports/keeper_pose_metrics.json
```

Ejemplos para detector de balon:

```text
precision
recall
mAP50
mAP50-95
validation_split
imgsz
conf_threshold
iou_threshold
```

La confianza de una observacion especifica si debe ir en la base:

```text
ball_confidence
shot_ball_confidence
keeper_confidence
head_confidence
left_hand_confidence
right_hand_confidence
left_foot_confidence
right_foot_confidence
keeper_pose_confidence
keeper_polygon_confidence
pose_valid_points_count
```

Regla:

```text
metrica global del modelo -> outputs/model_reports/
confianza de una deteccion/fila/jugada -> CSV frame-level o shot-level
```
