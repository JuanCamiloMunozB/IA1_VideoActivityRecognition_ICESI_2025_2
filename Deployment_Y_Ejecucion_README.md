# Ejecutable de Reconocimiento de Actividades - Video Activity Recognition

## 🎯 Inicio Rápido

### Windows Ejecutar la Aplicación 

Descargar aplicacion desde 
https://drive.google.com/file/d/1FdMknTccJYTd1WjFiD7QRudbJ90btPzD/view?usp=sharing

Ejecutarla con el siguiente comando:
```bash
.\VideoActivityRecognition.exe
```

O simplemente haz **doble clic** en `VideoActivityRecognition.exe`

## ⚠️ Solución de Problemas

### "Windows protegió tu PC"

Este mensaje aparece porque el ejecutable no está firmado digitalmente.

**Solución:**
1. Haz clic en "Más información"
2. Haz clic en "Ejecutar de todas formas"

### No se puede acceder a la cámara

**Solución:**
1. Ve a **Configuración de Windows** → **Privacidad** → **Cámara**
2. Asegúrate de que "Permitir que las aplicaciones accedan a tu cámara" esté **activado**
3. Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc.)


## Linux/Mac Ejecutar la Aplicación

Descargar imagen de docker desde

https://drive.google.com/file/d/1_Pw_a9y8ckdY8dzQPoiOT_nzlx2yHsVd/view?usp=sharing


Cargar la imagen de docker desde el archivo .tar
```bash
docker load -i video-har.tar
```
Habilitar permisos de X11
```bash
xhost +local:docker
```
Iniciar el contenedor. Verifique que dev/video0 este sea su camara web.

```bash
sudo docker run -it --rm \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --network host \
  video-activity-recognition:latest
```




