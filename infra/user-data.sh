#!/usr/bin/env bash
# Arranque de la instancia: instala MLflow, lo deja como servicio y arma el
# apagado por inactividad. Se ejecuta una sola vez, al crear la maquina.
set -euo pipefail
exec > >(tee /var/log/user-data.log) 2>&1

VERSION_MLFLOW="3.3.2"

apt-get update
apt-get install -y python3-venv python3-pip

install -d -o ubuntu -g ubuntu /opt/mlflow /opt/mlflow/artifacts /var/log/mlflow

sudo -u ubuntu python3 -m venv /opt/mlflow/venv
sudo -u ubuntu /opt/mlflow/venv/bin/pip install --upgrade pip
sudo -u ubuntu /opt/mlflow/venv/bin/pip install "mlflow==${VERSION_MLFLOW}"

# --serve-artifacts hace que los artefactos suban a traves del servidor.
# Sin esa bandera, un cliente remoto intentaria escribir en su propio disco
# y los artefactos nunca llegarian a la EC2.
cat > /etc/systemd/system/mlflow.service <<'UNIT'
[Unit]
Description=MLflow tracking server
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/mlflow
ExecStart=/opt/mlflow/venv/bin/mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////opt/mlflow/mlflow.db \
    --serve-artifacts \
    --artifacts-destination file:///opt/mlflow/artifacts \
    --gunicorn-opts "--access-logfile /var/log/mlflow/access.log"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# ---------------------------------------------------------------------------
# Apagado por inactividad
#
# Tres senales, en este orden:
#   1. tope duro por uptime  -> apaga siempre, pase lo que pase
#   2. sesion SSH abierta    -> cuenta como actividad, no apaga
#   3. peticiones a MLflow   -> mtime del log de acceso
#
# El tope duro existe porque la UI de MLflow refresca sola: una pestana
# olvidada mantiene vivo el log de acceso indefinidamente.
# ---------------------------------------------------------------------------
cat > /usr/local/bin/apagar-si-inactivo <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

INACTIVIDAD=${INACTIVIDAD:-10800}   # 3 h sin peticiones
TOPE_DURO=${TOPE_DURO:-43200}       # 12 h encendida, sin excepcion
LOG=/var/log/mlflow/access.log

uptime_s=$(cut -d. -f1 /proc/uptime)
if (( uptime_s > TOPE_DURO )); then
    logger -t apagado "tope duro de ${TOPE_DURO}s alcanzado, deteniendo"
    systemctl poweroff
    exit 0
fi

# Alguien conectado por SSH: puede estar entrenando desde la maquina.
if who | grep -q .; then
    exit 0
fi

ahora=$(date +%s)
ultima=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)

if (( ahora - ultima > INACTIVIDAD )); then
    logger -t apagado "sin peticiones a MLflow por ${INACTIVIDAD}s, deteniendo"
    systemctl poweroff
fi
SCRIPT
chmod +x /usr/local/bin/apagar-si-inactivo

cat > /etc/systemd/system/apagar-si-inactivo.service <<'UNIT'
[Unit]
Description=Detiene la instancia si MLflow lleva rato sin uso

[Service]
Type=oneshot
ExecStart=/usr/local/bin/apagar-si-inactivo
UNIT

cat > /etc/systemd/system/apagar-si-inactivo.timer <<'UNIT'
[Unit]
Description=Revisa inactividad cada 5 minutos

[Timer]
OnBootSec=15min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
UNIT

# Escotilla de escape: desactivar y reactivar el apagado a mano.
cat > /usr/local/bin/mantener-prendida <<'SCRIPT'
#!/usr/bin/env bash
sudo systemctl stop apagar-si-inactivo.timer
echo "Apagado automatico DESACTIVADO. La maquina no se apaga sola."
echo "Para volver a activarlo: reanudar-apagado"
SCRIPT

cat > /usr/local/bin/reanudar-apagado <<'SCRIPT'
#!/usr/bin/env bash
sudo systemctl start apagar-si-inactivo.timer
echo "Apagado automatico reactivado."
SCRIPT
chmod +x /usr/local/bin/mantener-prendida /usr/local/bin/reanudar-apagado

cat > /etc/update-motd.d/99-mlflow <<'MOTD'
#!/usr/bin/env bash
echo
echo "  MLflow: http://$(curl -s --max-time 2 ifconfig.me):5000"
if systemctl is-active --quiet apagar-si-inactivo.timer; then
    echo "  Apagado automatico: ACTIVO (3 h sin uso, tope duro 12 h)"
    echo "  Para desactivarlo: mantener-prendida"
else
    echo "  Apagado automatico: DESACTIVADO -- ojo con la factura"
fi
echo
MOTD
chmod +x /etc/update-motd.d/99-mlflow

systemctl daemon-reload
systemctl enable --now mlflow.service
systemctl enable --now apagar-si-inactivo.timer

echo "Listo. MLflow ${VERSION_MLFLOW} en el puerto 5000."
