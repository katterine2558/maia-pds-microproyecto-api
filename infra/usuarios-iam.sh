#!/usr/bin/env bash
# Crea los usuarios IAM del equipo.
#
#   katerine                    -> administrador (para dejar de usar root)
#   camilo, rainer, leonardo    -> solo pueden PRENDER la maquina de MLflow
#
# Todos entran por:  https://<CUENTA_AWS>.signin.aws.amazon.com/console
# Las contrasenas son temporales: AWS obliga a cambiarlas al primer ingreso.
#
# Correr una sola vez:  bash infra/usuarios-iam.sh
set -euo pipefail

# Valores concretos (ids de cuenta, IPs del equipo). No versionado:
# el repositorio es publico. Ver infra/valores.env.example
VALORES="$(dirname "$0")/valores.env"
if [[ ! -f "$VALORES" ]]; then
    echo "ERROR: falta infra/valores.env. Copialo de valores.env.example y llenalo." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VALORES"

CUENTA="$CUENTA_AWS"
EQUIPO=(camilo rainer leonardo)

clave() { LC_ALL=C tr -dc 'A-Za-z0-9!@#%^*_=+-' </dev/urandom | head -c 20; }

# Tolera que un recurso ya exista, para poder reintentar sin romper nada.
suave() { "$@" >/dev/null 2>&1 || true; }

echo "==> Politica prender-mlflow"
POLITICA=$(mktemp)
sed -e "s/CUENTA_AWS/${CUENTA}/" -e "s/INSTANCIA_ID/${INSTANCIA}/" \
    infra/iam-equipo.json > "$POLITICA"
suave aws iam create-policy \
    --policy-name prender-mlflow \
    --policy-document "file://${POLITICA}"
rm -f "$POLITICA"

echo "==> Grupo equipo-mlflow"
suave aws iam create-group --group-name equipo-mlflow
suave aws iam attach-group-policy --group-name equipo-mlflow \
    --policy-arn "arn:aws:iam::${CUENTA}:policy/prender-mlflow"
# Sin esta politica no pueden cambiar su propia contrasena al entrar.
suave aws iam attach-group-policy --group-name equipo-mlflow \
    --policy-arn arn:aws:iam::aws:policy/IAMUserChangePassword

declare -A CLAVES

echo "==> Usuarios del equipo"
for u in "${EQUIPO[@]}"; do
    suave aws iam create-user --user-name "$u"
    suave aws iam add-user-to-group --user-name "$u" --group-name equipo-mlflow
    c=$(clave); CLAVES[$u]=$c
    aws iam create-login-profile --user-name "$u" \
        --password "$c" --password-reset-required >/dev/null
    echo "    $u"
done

echo "==> Administrador"
suave aws iam create-user --user-name katerine
suave aws iam attach-user-policy --user-name katerine \
    --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
c=$(clave); CLAVES[katerine]=$c
aws iam create-login-profile --user-name katerine \
    --password "$c" --password-reset-required >/dev/null
echo "    katerine"

echo
echo "=============================================================="
echo " URL de ingreso"
echo "   https://${CUENTA}.signin.aws.amazon.com/console"
echo
echo " CONTRASENAS TEMPORALES (cambio obligatorio al primer ingreso)"
for u in "${!CLAVES[@]}"; do
    printf "   %-10s %s\n" "$u" "${CLAVES[$u]}"
done
echo "=============================================================="
echo
echo " Repartelas por un canal privado, no por el grupo del curso."
echo " Activa MFA en los cuatro usuarios, empezando por katerine."
echo " Despues de entrar como katerine, deja de usar la cuenta root."
