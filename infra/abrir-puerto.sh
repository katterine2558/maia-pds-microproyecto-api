#!/usr/bin/env bash
# Abre el puerto 5000 a internet para que el acceso a MLflow deje de depender
# de la IP desde la que se conecte cada integrante.
#
# Requisito previo: infra/mlflow-proxy.sh, que pone nginx delante pidiendo
# usuario y contrasena. Sin eso, esto dejaria el servidor abierto sin ninguna
# proteccion. El script lo verifica antes de tocar nada.
#
# El puerto 22 (SSH) NO se abre: sigue restringido a la IP de la duena.
#
# Correr:  bash infra/abrir-puerto.sh
set -euo pipefail

VALORES="$(dirname "$0")/valores.env"
if [[ ! -f "$VALORES" ]]; then
    echo "ERROR: falta infra/valores.env." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VALORES"

echo "==> Verificando que la autenticacion este activa"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
    "http://${IP_MLFLOW}:5000/health" || echo "000")

if [[ "$code" != "401" ]]; then
    echo "ABORTADO: sin credenciales el servidor responde HTTP ${code}, no 401." >&2
    echo "Abrir el puerto ahora dejaria MLflow expuesto sin proteccion." >&2
    echo "Corre primero: bash infra/mlflow-proxy.sh" >&2
    exit 1
fi
echo "    responde 401 sin credenciales, correcto"

echo "==> Abriendo el puerto 5000"
aws ec2 authorize-security-group-ingress \
    --region "$REGION" --group-id "$SECURITY_GROUP" \
    --ip-permissions 'IpProtocol=tcp,FromPort=5000,ToPort=5000,IpRanges=[{CidrIp=0.0.0.0/0,Description="MLflow tras nginx con autenticacion basica"}]' \
    --query 'SecurityGroupRules[0].SecurityGroupRuleId' --output text

echo "==> Quitando las reglas por IP en 5000, ya redundantes"
for cidr in $IPS_EQUIPO; do
    aws ec2 revoke-security-group-ingress \
        --region "$REGION" --group-id "$SECURITY_GROUP" \
        --protocol tcp --port 5000 --cidr "$cidr" \
        --query 'Return' --output text 2>/dev/null || true
done

echo "==> Reglas finales"
aws ec2 describe-security-groups --region "$REGION" --group-ids "$SECURITY_GROUP" \
    --query 'SecurityGroups[0].IpPermissions[].{puerto:FromPort,origen:IpRanges[].CidrIp}' \
    --output json

cat <<FIN

 MLflow queda accesible desde cualquier parte:

     http://${IP_MLFLOW}:5000

 SSH (puerto 22) sigue restringido a la IP de la duena. Si necesitas entrar
 desde otra red, hay que agregar esa IP a mano.
FIN
