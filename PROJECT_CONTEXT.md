# PROJECT_CONTEXT — autoclasificación jerárquica 2D/3D

Última actualización: 2026-07-24.

## Objetivo actual

Entender y estabilizar:

- `Prot2DAutoClassifier`
- `Prot3DAutoClassifier`
- la creación de sus salidas Scipion en `createOutputStep`
- los fallos intermitentes al insertar dichas salidas en SQLite

Este fichero es contexto vivo. Antes de usar conclusiones antiguas hay que
contrastar código, versiones, `git status`, Run fallido y traceback actuales.

## Fuentes analizadas

- Código:
  - `cryomethods/protocols/protocol_2d_auto_classifier.py`
  - `cryomethods/protocols/protocol_3d_auto_classifier.py`
  - `cryomethods/protocols/protocol_auto_base.py`
  - `cryomethods/protocols/protocol_base.py`
  - `cryomethods/functions.py`
  - `cryomethods/convert/`
- Implementación instalada de Scipion:
  - `pwem.objects.SetOfClasses.classifyItems`
  - `pwem.protocols.EMProtocol._createSetOfClasses*`
  - `pyworkflow.mapper.sqlite`
- Run 3D exitoso:
  - `/mnt/DATOS2/jvargas/ScipionUserData/projects/TestWorkflowRelionBetagal/Runs/003598_Prot3DAutoClassifier`
- Artículo:
  - [Hierarchical autoclassification of cryo-EM samples and macromolecular energy landscape determination](https://doi.org/10.1016/j.cmpb.2022.106673)

## Estado del checkout

- Rama: `devel`
- Commit inspeccionado: `a2665dab1ae21d5854d89eeac97919567ff0be2d`
- Plugin: `scipion-em-cryomethods 3.0.0`, editable.
- Había cambios previos del usuario; no se tocaron:
  - modificado `cryomethods/protocols/__init__.py`
  - modificado `cryomethods/viewers.py`
  - eliminados protocolos CTF y `cryomethods/scripts/script_CTF.py`
  - no versionado `prompt`

## Herencia real

La herencia actual no pasa por una clase de protocolo de
`scipion-em-relion`:

```text
Prot2DAutoClassifier / Prot3DAutoClassifier
    -> ProtAutoBase
        -> ProtocolBase
            -> pwem.protocols.EMProtocol
```

`ProtocolBase` reproduce/adapta lógica histórica de los protocolos Relion,
pero ejecuta directamente `relion_refine` o `relion_refine_mpi`. Relion sí es
el backend numérico.

Esto importa: arreglos recientes de `scipion-em-relion` no llegan
automáticamente a estos protocolos.

## Método descrito en el artículo

### 3D

1. Estimar una línea ResLog usando subconjuntos de partículas y
   clasificaciones Relion 3D con `K=1`.
2. Dividir jerárquicamente cada nodo en pocas clases, normalmente 2.
3. Continuar una rama si:
   - tiene suficientes partículas;
   - su calidad supera la esperada por ResLog.
4. Determinar automáticamente la convergencia mediante el plateau de
   `rlnAveragePmax`.
5. Opcionalmente alinear y agrupar mapas parecidos para obtener clases
   estructurales finales.

### 2D

- Misma división jerárquica.
- Relion 2D como backend.
- La continuación depende principalmente del número de partículas.
- No usa ResLog.

## Implementación actual

### Preparación general

`ProtAutoBase._insertAllSteps` inicializa estado dinámico:

- `_level`: nivel del árbol.
- `_rLev`: rama dentro del nivel.
- `stopDict`: mapa final/continuable.
- `stopResLog`: puntos ResLog.
- `_mapsDict`: mapa Relion -> identificador jerárquico.
- `_clsIdDict`: identificador jerárquico -> clase final consecutiva.
- `_evalIdsList` y `_doneList`: control de pasos dinámicos.

Si es 3D y `useReslog=True`, empieza en nivel 0. En otro caso empieza en
nivel 1.

### Ejecución Relion

Cada nodo ejecuta `relion_refine[_mpi]`.

- Clasificación jerárquica: `--K numberOfClasses`.
- Fase ResLog: fuerza `--K 1`.
- Primera ejecución: 5 iteraciones.
- Después comprueba la pendiente de `rlnAveragePmax` en seis iteraciones.
- Convergencia actual: pendiente `<= 0.005`.
- Si no converge, continúa en bloques de 5.

Diferencia importante con el artículo: el código no ejecuta literalmente
“10 iteraciones iniciales y máximo 75”. Empieza con 5 y el bucle llega como
máximo a 50 iteraciones.

### Evaluación de ramas

Tras cada nivel:

1. `_copyLevelMaps` copia clases con `rlnClassDistribution > 0.05`.
2. `_evalStop` calcula tamaño y condición de parada.
3. `_mergeModelStar` separa mapas terminales y continuables.
4. `_mergeDataStar` separa partículas terminales y continuables.
5. 3D calcula mapa promedio y alinea los mapas del nivel.

Una rama termina si:

- `classSize < minPartsToStop`; o
- el criterio ResLog falla; o
- la clase contiene al menos aproximadamente el 95 % del nodo padre.

Clases Relion con distribución `<= 0.05` no se conservan como mapa. Sus
partículas se reasignan a una clase retenida mediante el bucle de fallback de
`_mergeDataStar`. Esto debe considerarse una decisión/limitación del algoritmo,
no un detalle SQLite.

### Agrupamiento 3D opcional

Con `doGrouping=False`, `raw_final_*` se copia a `final_*`.

Con `doGrouping=True`:

- se obtienen vectores de los mapas en `NumpyImgHandler.getAllNpList`;
- se agrupan por K-means o Affinity Propagation;
- se vuelve a ejecutar una clasificación `K=1` por grupo;
- se generan `final_data.star` y `final_model.star`.

El default es Affinity Propagation. La ruta ejecutada por
`mergeClassesStep` no aplica explícitamente la tPCA del 90 % descrita en el
artículo. Además, el K-means actual usa `matProj.shape[1]` como número de
clusters, lo que parece incorrecto si esa dimensión es el número de voxeles.

## Creación de salida Scipion

### Flujo común

`_fillClassesFromIter`:

1. Lee `final_model.star`.
2. Enumera sus filas como clases `1..N`.
3. Lee `final_data.star` ordenado por `rlnImageId`.
4. Itera las partículas de entrada por `id`.
5. Clona cada partícula (`doClone=True`).
6. Actualiza clase, transformación y metadatos Relion.
7. `SetOfClasses.classifyItems` inserta:
   - clases en `Objects`;
   - partículas en tablas `ClassNNN_Objects`;
   - esquemas en las tablas `Classes`.

La ordenación solo es correcta si los IDs de las partículas de entrada y de
`final_data.star` son el mismo conjunto.

### 2D

`createOutputStep` crea:

- `classes2D.sqlite`
- salida `outputClasses`
- relación de procedencia desde `inputParticles`

### 3D

`createOutputStep` crea:

- `classes3D.sqlite`
- salida `outputClasses`
- `volumes.sqlite`
- salida `outputVolumes`
- relaciones desde partículas y volumen inicial

Cada volumen de salida reutiliza el representante de su `Class3D` y adopta el
mismo `objId`.

### Cambio histórico relevante

Commit `d05ceee` (2024-06-03) cambió:

```python
doClone=False -> doClone=True
```

El propósito era evitar problemas al registrar partículas. Hay que conservar
este dato al comparar Runs antiguos y nuevos.

## Run 3D exitoso `003598`

### Entorno

- Inicio: 2026-07-24 10:52:08.
- Fin: 2026-07-24 11:13:48.
- Relion ejecutado: `4.0.0-commit-138b9c`.
- `scipion-pyworkflow`: 3.11.6.
- `scipion-em`: 3.11.0.
- `scipion-em-relion`: 4.0.11.
- SQLite journal global: `DELETE`.
- Proyecto sobre filesystem local `ext4`.

El código declara soporte histórico para Relion 3.0/3.1, pero este Run usa
Relion 4.0.

### Parámetros principales

- Entrada: 3139 partículas de
  `001578_ProtRelionRefine3D/outputParticles`.
- Pixel: 3.54 Å/px.
- Caja: 64 px.
- Volumen inicial: `001578_ProtRelionRefine3D/outputVolume`.
- Simetría: `d2`.
- `numberOfClasses=2`.
- `minPartsToStop=500`.
- `useReslog=True`.
- `doGrouping=False`.
- `regularisationParamT=4`.
- alineamiento activo.
- angular sampling enum `1`.
- offsets: rango 5 px, paso 1 px.
- MPI 3, threads 2, GPU activo.

### Árbol ejecutado

- ResLog: nivel 0, ramas 2..9.
- Clasificación: niveles 1..6.
- Se generaron 8 clases terminales.
- `createOutputStep`: 1.32 s.

### Artefactos finales verificados

- `final_data.star`: 3139 filas.
- IDs STAR: 3139 únicos.
- Diferencias entre IDs STAR e IDs de entrada: 0.
- `final_model.star`: 8 clases.
- `classes3D.sqlite`: `PRAGMA integrity_check = ok`.
- `volumes.sqlite`: `PRAGMA integrity_check = ok`.
- No quedaron ficheros `-journal`, `-wal` ni `-shm`.

Partículas por clase:

| Clase | Partículas |
|---:|---:|
| 1 | 480 |
| 2 | 375 |
| 3 | 300 |
| 4 | 174 |
| 5 | 766 |
| 6 | 227 |
| 7 | 197 |
| 8 | 620 |
| Total | 3139 |

Se reprodujo en `/tmp` la creación real de `classes3D.sqlite` usando las 3139
partículas y ambos STAR del Run. Resultado: 20/20 creaciones correctas. Por
tanto, los artefactos del Run exitoso no contienen un fallo determinista de
inserción.

## Hallazgo independiente: ResLog está roto en este Run

En nivel 0, `resLogStep` hace:

```python
mdData = self._getMetadata(imgStar)
size = np.math.log10(mdData.size())
```

Con STAR de Relion 4, leer el fichero sin especificar `particles@` toma el
bloque de óptica, cuyo tamaño es 1. Consecuencia observada:

- todos los puntos usan clave `log10(1) = 0.0`;
- cada punto sobrescribe al anterior en `stopResLog`;
- `_getReslogVars` devuelve pendiente, intercepto y error `nan`;
- el log muestra `EvalStop mx+n: m: nan, n nan, err nan`.

Así, la condición ResLog no está funcionando. La jerarquía de este Run se
detuvo básicamente por tamaño y por el criterio del 95 %.

Arreglo probable, todavía no implementado:

```python
mdData = self._getMetadata('particles@' + imgStar)
```

Hay que verificar después que las ocho muestras generen ocho valores de
`log10(N)` distintos y una regresión finita.

## Estado del fallo SQLite

El Run proporcionado es sano. No contiene el traceback del fallo aleatorio.
Sin la excepción exacta no se puede afirmar la causa.

Hipótesis a distinguir:

1. `database is locked` / `database table is locked`:
   concurrencia o transacción sin cerrar.
2. `UNIQUE constraint failed ... Objects.id`:
   ID duplicado o reuso incorrecto.
3. `StopIteration` o error durante `_updateParticle`:
   desajuste entre partículas de entrada y filas STAR; puede dejar SQLite
   parcial aunque SQLite no sea la causa primaria.
4. `Error binding parameter` / tipo no soportado:
   metadato Relion no normalizado antes de insertarlo.
5. `disk I/O error`, `readonly database`, cuota o permisos:
   problema de filesystem.
6. Fallo al insertar `outputClasses`/relaciones en `run.db`, no al rellenar
   `classes3D.sqlite`.

### Debilidad actual

`createOutputStep` no valida antes de escribir:

- número de partículas;
- unicidad e igualdad de IDs;
- clases usadas en data frente a clases del model;
- columnas Relion requeridas;
- integridad del SQLite recién generado.

Un error de STAR puede aparecer tarde, durante una inserción, y confundirse
con un error de base de datos.

## Siguiente diagnóstico mínimo

Necesario: ruta del Run fallido completo o, como mínimo:

- `logs/run.stdout`
- `logs/run.stderr`
- traceback mostrado por Scipion
- `logs/run.db`
- `extra/final_data.star`
- `extra/final_model.star`
- SQLite parcial, si existe

No relanzar `createOutputStep` antes de copiar el SQLite parcial:
`_createSetOfClasses*` elimina el fichero de salida anterior.

Comprobaciones iniciales:

```bash
rg -n -i -C 15 \
  'createOutput|traceback|sqlite|database|integrity|operationalerror|error' \
  RUN/logs/run.stdout RUN/logs/run.stderr

sqlite3 RUN/classes3D.sqlite 'PRAGMA integrity_check;'
sqlite3 RUN/volumes.sqlite 'PRAGMA integrity_check;'
```

## Estrategia de corrección

Primero reproducir y clasificar la excepción. Después:

1. Añadir una validación previa STAR/entrada.
2. Marcar en log cada subfase de `createOutputStep`.
3. Hacer commit explícito con `classes.write()` y `volumes.write()` antes de
   registrar las salidas.
4. Cerrar explícitamente sets/mappers cuando ya no se usen.
5. Añadir un test de creación repetida desde STAR conocidos.
6. Solo si el error exacto es un bloqueo, evaluar retry localizado o WAL.
   No cambiar el journal global de Scipion a ciegas.

## Riesgos y deuda técnica relacionados

- Compatibilidad declarada Relion 3.x frente a ejecución real Relion 4.
- ResLog roto por selección incorrecta del bloque STAR.
- Máximo de iteraciones distinto del artículo.
- Clases `<=5 %` se descartan y sus partículas se reasignan.
- K-means opcional parece usar un número de clusters incorrecto.
- El aviso de Scipion sobre puntero directo aparece porque se pasa
  `inputParticles.get()` a `_createSetOfClasses*`; no fue fatal en el Run
  exitoso, pero conviene modernizarlo.
- La corrección SQLite debe mantenerse separada de la corrección científica
  de ResLog para poder validar cada cambio.

