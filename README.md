# 🚛 SmartDispatch

https://smartdispatch-czgxdrvx6ifwnb8svruz3j.streamlit.app/

### Optimizador Híbrido de Flota para Logística de Construcción

SmartDispatch resuelve un problema real de logística: **asignar pedidos de materiales de construcción a camiones, generando la hoja de ruta óptima del día** de forma automática.

El sistema combina dos técnicas de IA:

- **Algoritmo Genético (AG)** — encuentra la distribución de pedidos en camiones que minimiza la distancia total recorrida.
- **Sistema Experto (SE)** — valida cada solución contra 30 reglas de negocio: incompatibilidades químicas, límites legales de peso, ventanas horarias de entrega y certificaciones de proveedores.

La integración es lo que distingue al sistema: el SE actúa dentro de la función de fitness del AG, penalizando las soluciones inviables para que sean eliminadas evolutivamente. El resultado es una ruta que es simultáneamente **óptima en costo y 100% válida** según las restricciones del negocio.

---

## El problema que resuelve

Planificar manualmente qué material va en qué camión y en qué orden toma entre 3 y 5 horas. Un error de peso genera una multa de tránsito. Mezclar cemento con yeso en el mismo camión provoca fraguado y arruina la carga. Llegar tarde a una obra detiene a toda la cuadrilla.

SmartDispatch genera la hoja de ruta completa en menos de 60 segundos.

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Algoritmo Genético | DEAP (Python) |
| Sistema Experto | Forward Chaining (implementación propia) |
| Interfaz web | Streamlit |
| Mapas interactivos | Folium + streamlit-folium |
| Visualizaciones | Plotly |
| Procesamiento de datos | Pandas · NumPy |

---

## Cómo correr la app

```bash
pip install -r requirements.txt
streamlit run app.py
```

O accedé directamente a la versión deployada en Streamlit Cloud.


---

## Estructura de la app

| Pestaña | Contenido |
|---|---|
| 🏠 Inicio | Descripción del sistema y arquitectura híbrida |
| 📊 EDA | Análisis exploratorio del dataset: gráficos y mapa de obras |
| 🧠 Sistema Experto | Base de conocimiento (30 reglas) y simulador interactivo |
| 🧬 Optimización AG | Configuración y ejecución del algoritmo con gráfico de convergencia |
| 📋 Resultados | Hoja de ruta final con auditoría de compliance por camión y exportación CSV |

---

## Equipo

| Nombre | 
|---|
| Gustavo Benitez |
| Nazareth Vargas |
| Frank Gonzalez |
| Carla Paz |
| Bautista Leon |

---

*Modelizado de Sistemas de IA*
