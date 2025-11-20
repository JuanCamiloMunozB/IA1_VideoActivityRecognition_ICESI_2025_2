[LOG - Estado actual del sistema HAR (Entrega 3)]
- Actualización realtime_inference.py:
    - Los vectores que se le pasan al modelo para que este haga su predicción ya no corresponden a frames consecutivos. Ahora, se le envia información cada 6 frames. Fue así tambien como se procesaron los videos en el entrenamiento. 
    - El desempeño de la clase pararse y sentarse mejoró significativamente. La clase girar sigue siendo problemática.