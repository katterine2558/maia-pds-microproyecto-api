#!/usr/bin/env bash
# Alarma de costo: avisa por correo cuando EC2 + Elastic IP + disco se acerquen
# a USD 5 en el mes.
#
# Sin filtros: cubre el gasto TOTAL de la cuenta, no solo EC2. Si manana se
# suma S3, un balanceador o lo que sea, entra en la misma cuenta de USD 5.
#
# Correr una sola vez. Requiere que Cost Explorer este habilitado en la cuenta
# (Billing -> Cost Explorer -> Enable). La primera habilitacion tarda ~24 h en
# mostrar datos, pero el presupuesto se crea de una.
set -euo pipefail

LIMITE="5"

# Valores concretos (ids de cuenta, IPs del equipo). No versionado:
# el repositorio es publico. Ver infra/valores.env.example
VALORES="$(dirname "$0")/valores.env"
if [[ ! -f "$VALORES" ]]; then
    echo "ERROR: falta infra/valores.env. Copialo de valores.env.example y llenalo." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VALORES"

CORREO="${1:-$CORREO_ALERTAS}"
CUENTA="$CUENTA_AWS"
TMP=$(mktemp -d)

cat > "$TMP/presupuesto.json" <<JSON
{
  "BudgetName": "maia-pds-total",
  "BudgetLimit": { "Amount": "${LIMITE}", "Unit": "USD" },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
JSON

# Tres avisos: al 60% y 80% del gasto real, y cuando la proyeccion del mes
# supere el limite. El de proyeccion es el que avisa a tiempo.
cat > "$TMP/notificaciones.json" <<JSON
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 60,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "${CORREO}" }]
  },
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "${CORREO}" }]
  },
  {
    "Notification": {
      "NotificationType": "FORECASTED",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 100,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{ "SubscriptionType": "EMAIL", "Address": "${CORREO}" }]
  }
]
JSON

aws budgets create-budget \
    --account-id "$CUENTA" \
    --budget "file://$TMP/presupuesto.json" \
    --notifications-with-subscribers "file://$TMP/notificaciones.json"

rm -rf "$TMP"

cat <<RESUMEN

Presupuesto "maia-pds-total" creado: USD ${LIMITE}/mes sobre TODA la cuenta.
Avisos a ${CORREO} al 60%, al 80% y cuando la proyeccion pase los ${LIMITE}.

Los avisos llegan directo al correo, sin paso de confirmacion (eso solo
aplica cuando el destino es un topico SNS).
RESUMEN
