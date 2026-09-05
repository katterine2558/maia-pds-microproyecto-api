#!/usr/bin/env bash
# Pone nginx delante de MLflow con autenticacion basica, para que el acceso no
# dependa de la IP desde la que se conecte cada integrante.
#
# Por que no la autenticacion propia de MLflow: guarda las claves con scrypt,
# que reserva ~32 MB y tarda ~122 ms POR PETICION. Con 4 workers en paralelo
# sobre una maquina de 1 GB, el kernel mata el proceso. Medido: 16 muertes por
# OOM en 30 minutos. nginx con bcrypt de coste 5 gasta ~4 KB y ~5 ms.
#
# Reparto de puertos despues de esto:
#   nginx   0.0.0.0:5000  <- publico, pide usuario y clave
#   mlflow  127.0.0.1:5001 <- solo local, sin autenticacion propia
#
# Correr:  bash infra/mlflow-proxy.sh
set -euo pipefail

VALORES="$(dirname "$0")/valores.env"
if [[ ! -f "$VALORES" ]]; then
    echo "ERROR: falta infra/valores.env." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VALORES"

LLAVE="${HOME}/.ssh/mlflow-maia.pem"
USUARIOS="katerine camilo rainer leonardo"

ssh -i "$LLAVE" -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
    "ubuntu@${IP_MLFLOW}" "USUARIOS='${USUARIOS}' bash -s" <<'REMOTO'
set -euo pipefail

echo "==> Instalando nginx"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nginx apache2-utils >/dev/null

echo "==> MLflow vuelve a local, sin su autenticacion"
# Se reescribe la unidad completa en vez de parcharla: quedaba con restos de
# los intentos anteriores.
sudo tee /etc/systemd/system/mlflow.service >/dev/null <<'UNIT'
[Unit]
Description=MLflow tracking server
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/opt/mlflow
ExecStart=/opt/mlflow/venv/bin/mlflow server \
    --host 127.0.0.1 \
    --port 5001 \
    --workers 2 \
    --backend-store-uri sqlite:////opt/mlflow/mlflow.db \
    --serve-artifacts \
    --artifacts-destination file:///opt/mlflow/artifacts
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

echo "==> Usuarios de nginx (bcrypt, coste 5)"
CLAVES=""
primero=si
for u in $USUARIOS; do
    c=$(python3 -c "import secrets, string
alfabeto = string.ascii_letters + string.digits
print(''.join(secrets.choice(alfabeto) for _ in range(16)))")
    if [[ "$primero" == "si" ]]; then
        sudo htpasswd -B -C 5 -b -c /etc/nginx/mlflow.htpasswd "$u" "$c" >/dev/null 2>&1
        primero=no
    else
        sudo htpasswd -B -C 5 -b /etc/nginx/mlflow.htpasswd "$u" "$c" >/dev/null 2>&1
    fi
    CLAVES="${CLAVES}${u} ${c}"$'\n'
done
sudo chmod 640 /etc/nginx/mlflow.htpasswd
sudo chown root:www-data /etc/nginx/mlflow.htpasswd

echo "==> Sitio de nginx"
sudo tee /etc/nginx/sites-available/mlflow >/dev/null <<'SITIO'
server {
    listen 5000;
    server_name _;

    access_log /var/log/nginx/mlflow.access.log;

    # Los artefactos pueden ser grandes y algunas llamadas tardan.
    client_max_body_size 512M;
    proxy_read_timeout 300s;

    location / {
        auth_basic "MLflow - micro-proyecto MAIA";
        auth_basic_user_file /etc/nginx/mlflow.htpasswd;

        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
SITIO

sudo ln -sf /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/mlflow
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t

echo "==> El apagado por inactividad ahora mira el log de nginx"
sudo sed -i 's|^LOG=/var/log/mlflow/access.log|LOG=/var/log/nginx/mlflow.access.log|' \
    /usr/local/bin/apagar-si-inactivo

echo "==> Reiniciando"
sudo systemctl daemon-reload
sudo systemctl restart mlflow.service
sudo systemctl restart nginx
sudo systemctl enable nginx >/dev/null 2>&1

for _ in $(seq 1 25); do
    curl -s -o /dev/null --max-time 3 http://127.0.0.1:5001/health && break
    sleep 3
done

primer_usuario=$(echo "$CLAVES" | head -1 | cut -d' ' -f1)
primera_clave=$(echo "$CLAVES" | head -1 | cut -d' ' -f2)

echo "==> Verificacion"
sin=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:5000/health || echo "000")
con=$(curl -s -o /dev/null -w '%{http_code}' -u "${primer_usuario}:${primera_clave}" \
      --max-time 10 http://localhost:5000/health || echo "000")
echo "    sin credenciales -> HTTP ${sin}   (debe ser 401)"
echo "    con usuario      -> HTTP ${con}   (debe ser 200)"

echo "==> Coste de una verificacion bcrypt"
python3 - <<'MEDIR'
import subprocess, time
t = time.time()
subprocess.run(["htpasswd", "-bnB", "-C", "5", "x", "y"],
               capture_output=True, text=True)
print(f"    {(time.time()-t)*1000:.0f} ms  (scrypt de MLflow: 122 ms)")
MEDIR

echo
echo "=============================================================="
echo " CREDENCIALES DE MLFLOW"
echo
echo "$CLAVES" | while read -r u c; do
    [[ -n "$u" ]] && printf "   %-10s %s\n" "$u" "$c"
done
echo "=============================================================="
REMOTO

cat <<'FIN'

 Falta abrir el puerto 5000 a internet para que el acceso deje de depender
 de la IP. Eso es un paso aparte y va despues de confirmar el 401.

 OJO: sobre HTTP plano las credenciales viajan en base64, sin cifrar. Con el
 puerto abierto, cualquiera en la ruta de red puede leerlas. Que nadie
 reutilice ahi una contrasena de otro servicio. La solucion definitiva es
 TLS con un nombre de dominio, que se puede montar despues de la entrega.
FIN
