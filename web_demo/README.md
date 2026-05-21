# Keeper Probability Demo UI

Este demo muestra un mapa interactivo de `P(save)`: la probabilidad estimada de
que el portero pare un tiro dependiendo del punto donde el balon llega a la
portería.

## Como correrlo

Desde la raiz del repo:

```bash
python -m http.server 8010 --directory web_demo
```

Luego abre:

```text
http://127.0.0.1:8010/keeper_probability_demo.html
```

Esto es preferible a abrir el HTML con `file://`, porque el servidor local deja
que el demo cargue `save_probability_model.json`, que contiene los coeficientes
del modelo `Baseline ML (PFF)`.

## Que significa cada cosa

### Punto blanco / objetivo

El punto blanco dentro de la porteria es el punto de entrada o llegada del tiro.
En la base de datos corresponde a:

```text
goal_entry_u
goal_entry_v
```

`goal_entry_u` mide posicion horizontal:

```text
0.00 = poste izquierdo
0.50 = centro
1.00 = poste derecho
```

`goal_entry_v` mide altura:

```text
0.00 = suelo
0.50 = media altura
1.00 = travesano
```

La tarjeta `P parada objetivo` arriba muestra la probabilidad de parada
exactamente en ese punto blanco.

### Heatmap

Cada color del arco representa la probabilidad estimada si el balon llegara a
ese punto:

```text
rojo/ocre = menor probabilidad de parada
verde/azul = mayor probabilidad de parada
```

Las tarjetas `P >= 0.80` y `P >= 0.50` indican que porcentaje del arco queda por
encima de esos umbrales.

La vista del campo queda fija en el zoom minimo para que el demo muestre toda el
area grande, el area chica, el punto penal, el portero y el balon en un mismo
contexto espacial. El punto amarillo indica la posicion original del tiro; el
punto blanco dentro del arco indica el objetivo o entrada estimada del balon.

### Incertidumbre y decisión

El panel `Incertidumbre y decisión` agrega tres conceptos:

```text
Monte Carlo      simula tiros cercanos al punto blanco
Entropia         mide incertidumbre en bits
Optimizacion     aproxima la mejor posicion valida del portero
```

La media Monte Carlo resume la probabilidad esperada si el tiro no llega
exactamente al punto blanco, sino a una nube de puntos cercanos. El intervalo se
controla con `alpha`:

```text
alpha = 0.10 -> IC 90%  -> percentiles 5 y 95
alpha = 0.05 -> IC 95%  -> percentiles 2.5 y 97.5
alpha = 0.01 -> IC 99%  -> percentiles 0.5 y 99.5
```

Este `alpha` solo cambia el rango de incertidumbre reportado; no cambia ni
infla la probabilidad media del modelo.

La barra `Incertidumbre` tampoco debe interpretarse como una variable que sube o
baja la probabilidad por si sola. En esta version controla la dispersion de la
nube Monte Carlo: a mayor incertidumbre, se simulan tiros mas dispersos alrededor
del objetivo y por eso puede cambiar el ancho del intervalo. La media del modelo
se mantiene como salida de `P(save)` para evitar inflarla artificialmente.

La entropia se calcula como:

```text
H(p) = -p log2(p) - (1-p) log2(1-p)
```

Si `p = 0.5`, la incertidumbre es maxima (`1 bit`). Si `p` esta cerca de `0` o
`1`, la incertidumbre baja.

La mejor posicion del portero se calcula con una busqueda discreta refinada:
primero evalua una malla amplia dentro de la zona valida del modo activo y
despues hace una segunda busqueda local alrededor del mejor punto encontrado. El
boton `Aplicar optimo` mueve el portero a esa posicion sugerida.

El panel tambien incluye tres visualizaciones:

```text
Histograma MC       distribucion de probabilidades simuladas
Barra de entropia   incertidumbre del objetivo entre 0 y 1 bit
Actual vs optimo    comparacion de probabilidad esperada del portero actual
                    contra la posicion sugerida por busqueda refinada
```

### Resultados interpretables

El panel `Resultados interpretables del modelo` resume el resultado de forma
defendible para presentacion:

```text
Fuente / AUC / Brier / n    calidad y origen del modelo cargado
Lectura narrativa           interpretacion del tiro actual
Top factores actuales       variables con mayor contribucion al logit del ML
Conceptos teoricos          ML, Monte Carlo, entropia y optimizacion
```

Los factores positivos empujan la prediccion hacia mayor probabilidad de
parada; los negativos la empujan hacia menor probabilidad. Esta explicacion
aplica al `Baseline ML (PFF)`. En el `Simulador geometrico` no hay coeficientes
aprendidos, por lo que se muestra una nota aclarando que la salida viene de
supuestos de alcance y tiempo.

La `Guia de barras` es un menu desplegable para no ocupar espacio durante la
demostracion.

## Modos de modelo

### ML PFF

Usa una regresion logistica entrenada con el CSV generado desde PFF World Cup
2022:

```text
data/processed/pff_worldcup_2022_save_probability.csv
```

El modelo actual usa tiros etiquetados como:

```text
save_label = 1 -> parada
save_label = 0 -> gol
```

En este modo la UI solo muestra controles que tienen peso distinto de cero en el
modelo cargado.

Tambien restringe la posicion del portero a una zona plausible frente al arco:
`Centro U` queda entre postes (`0` a `1`) y `Centro V` queda dentro de la
profundidad del area chica. Esto evita que artefactos del modelo event-level,
como un peso positivo en `keeper_outside_goal`, hagan parecer que salir de la
zona defendible aumenta la probabilidad de parada.

Controles activos en `ML PFF`:

```text
Centro U / Centro V        posicion del portero
Posicion U / Posicion V    posicion del balon al tiro
Objetivo U / Objetivo V    punto de entrada del tiro
Tiempo al arco             tiempo estimado hasta porteria
Incertidumbre              dispersion de tiros Monte Carlo alrededor del objetivo
```

Controles ocultos en `ML PFF`:

```text
Alcance lateral
Alcance vertical
Deformacion
Velocidad
Reaccion
Tanda
```

Estan ocultos porque, con el dataset PFF actual, esas columnas no tienen
variacion real suficiente o tienen coeficiente cero en el modelo entrenado.

### Geom

Es una funcion geometrica proxy, no entrenada. Sirve para explicar el concepto
de alcance corporal/deformacion:

```text
mas cerca del cuerpo del portero -> mayor P(save)
mas tiempo al arco -> mayor P(save)
mas velocidad o menos reaccion -> menor P(save)
mas deformacion/alcance -> mayor zona cubierta
```

En `Geom` aparecen los controles de alcance, deformacion, velocidad y reaccion
porque ahi si modifican la funcion proxy.

## Por que a veces parece que no cambia

Puede pasar por tres razones:

1. El valor estaba redondeado. Ahora se muestra como porcentaje con un decimal
   para que los cambios pequenos sean visibles.
2. El modelo PFF actual es debil/modesto (`AUC` alrededor de `0.545`), por lo
   que algunas variables cambian la probabilidad muy poco.
3. Algunas variables no fueron aprendidas por PFF y por eso se ocultaron en
   `ML PFF`.

## Como usarlo en la presentacion

Flujo sugerido:

1. Modo `ML PFF`: explicar que aqui se usan datos reales de tiros del Mundial
   2022, pero con limitaciones.
2. Mover el punto blanco dentro del arco y mostrar como cambia `P parada
   objetivo`.
3. Usar el panel de incertidumbre para explicar Monte Carlo, entropia y
   optimizacion de posicion del portero.
4. Usar `Resultados interpretables` para explicar metricas, limitaciones y
   drivers del modelo.
5. Mover el portero y el balon para mostrar sensibilidad del modelo.
6. Cambiar a `Geom` para explicar el siguiente paso del proyecto: alcance
   corporal, deformacion y reaccion.

## Limitaciones actuales

- PFF entrega event-level broadcast tracking, no pose corporal completa.
- El poligono deformable todavia no esta entrenado con puntos corporales reales.
- La velocidad/reaccion/deformacion en `Geom` son demostrativas.
- El modelo `ML PFF` es una primera linea base, no un modelo final de decision
  del portero.
