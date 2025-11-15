[LOG - Estado actual del sistema HAR (Entrega 3)]

- Versión de despliegue:
  - Modelo principal: SVM reducido con selección de características (K=70).
  - Los artefactos (modelo, encoder de etiquetas, configuración de features) se cargan desde Entrega3/experiments/models y Entrega2/experiments/results.
  - Si el modelo reducido no está disponible, el sistema cae de forma automática al modelo "full".

- Pipeline en tiempo real:
  - La cámara se inicializa con el índice configurado (por defecto 0).
  - Se estima el FPS de la cámara; si no está disponible se usa un valor por defecto de 30 FPS.
  - Cada frame se procesa con MediaPipe Pose para extraer 33 landmarks corporales estándar y un landmark sintético de "head".
  - Se construye un buffer deslizante de frames y, cuando hay suficientes muestras, se genera un vector de características siguiendo la misma lógica de la Entrega 2.
  - El vector de características se pasa al pipeline de scikit-learn (SelectKBest + StandardScaler + SVM) y se obtiene:
    - La actividad predicha.
    - La probabilidad (o score normalizado) asociada a la clase más probable.

- Interfaz de usuario (UI):
  - Se muestra el video en vivo con un panel informativo superpuesto:
    - Nombre de la variante de modelo en uso (reduced/full) y FPS estimado.
    - Actividad detectada y nivel de confianza, con un código de color:
      - Verde: alta confianza (> 70%).
      - Amarillo: confianza media (40–70%).
      - Rojo: baja confianza (< 40%).
    - Visibilidad media de la pose y una advertencia si es baja.
    - Métricas de postura en tiempo real:
      - Inclinación del tronco (grados).
      - Ángulo de rodilla izquierda (grados).
      - Ángulo de rodilla derecha (grados).
  - En la parte inferior se muestra la instrucción “Pulsa 'q' para salir”.
  - La aplicación se cierra de forma controlada al presionar la tecla 'q' o si se pierde la señal de la cámara.

- Limitaciones conocidas (estado actual):
  - El sistema funciona de manera estable para actividades con patrones de movimiento más marcados y bien diferenciados.
  - Sin embargo, aún presenta dificultades para identificar correctamente algunas actividades que implican transiciones sutiles, especialmente:
    - Sentarse.
    - Levantarse desde la posición sentada.
  - En estas transiciones, la probabilidad puede ser baja o el modelo puede confundir la actividad con otras posturas estáticas, lo que indica que todavía hay margen de mejora en:
    - El diseño de las características.
    - El balance de datos de entrenamiento para estas clases.
    - O el ajuste fino del modelo SVM para dichas transiciones.