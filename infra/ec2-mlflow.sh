#!/usr/bin/env bash
# Crea la EC2 que hospeda el tracking server de MLflow.
#
# Idempotente a medias: si algo falla a mitad, revisa que recurso quedo creado
# antes de volver a correrlo. Cada paso imprime el id de lo que crea.
#
# Requisitos: aws cli v2 configurada con un usuario IAM administrador
# (NO la cuenta root) y permisos sobre EC2.
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

NOMBRE="mlflow-maia"
TIPO="t3.micro"
DISCO_GB=10

read -ra IPS <<< "$IPS_EQUIPO"
if (( ${#IPS[@]} == 0 )); then
    echo "ERROR: IPS_EQUIPO vacio en infra/valores.env" >&2
    exit 1
fi

echo "==> Ubuntu 24.04 LTS mas reciente"
AMI=$(aws ssm get-parameters --region "$REGION" \
    --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query 'Parameters[0].Value' --output text)
echo "    AMI: $AMI"

echo "==> Par de llaves"
if [[ ! -f "$HOME/.ssh/${NOMBRE}.pem" ]]; then
    aws ec2 create-key-pair --region "$REGION" \
        --key-name "$NOMBRE" \
        --query 'KeyMaterial' --output text > "$HOME/.ssh/${NOMBRE}.pem"
    chmod 400 "$HOME/.ssh/${NOMBRE}.pem"
    echo "    Guardada en ~/.ssh/${NOMBRE}.pem -- NO la subas al repo ni la compartas"
else
    echo "    Ya existe ~/.ssh/${NOMBRE}.pem"
fi

echo "==> Security group"
VPC=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters Name=is-default,Values=true \
    --query 'Vpcs[0].VpcId' --output text)

SG=$(aws ec2 create-security-group --region "$REGION" \
    --group-name "$NOMBRE" \
    --description "MLflow tracking server - micro-proyecto MAIA PDS" \
    --vpc-id "$VPC" \
    --query 'GroupId' --output text)
echo "    $SG"

for ip in "${IPS[@]}"; do
    for puerto in 22 5000; do
        aws ec2 authorize-security-group-ingress --region "$REGION" \
            --group-id "$SG" --protocol tcp --port "$puerto" --cidr "$ip" >/dev/null
    done
done
echo "    Puertos 22 y 5000 abiertos solo a las ${#IPS[@]} IPs del equipo"

echo "==> Instancia"
# instance-initiated-shutdown-behavior=stop es critico: el apagado automatico
# usa 'poweroff' desde dentro. Con 'terminate' la maquina se destruiria sola,
# y el enunciado exige dejarla detenida, no terminada.
ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$TIPO" \
    --key-name "$NOMBRE" \
    --security-group-ids "$SG" \
    --instance-initiated-shutdown-behavior stop \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DISCO_GB},VolumeType=gp3}" \
    --user-data "file://$(dirname "$0")/user-data.sh" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NOMBRE}},{Key=Proyecto,Value=maia-pds}]" \
    --query 'Instances[0].InstanceId' --output text)
echo "    $ID"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"

echo "==> Elastic IP"
ALLOC=$(aws ec2 allocate-address --region "$REGION" --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=${NOMBRE}},{Key=Proyecto,Value=maia-pds}]" \
    --query 'AllocationId' --output text)

aws ec2 associate-address --region "$REGION" \
    --instance-id "$ID" --allocation-id "$ALLOC" >/dev/null

IP=$(aws ec2 describe-addresses --region "$REGION" \
    --allocation-ids "$ALLOC" --query 'Addresses[0].PublicIp' --output text)

CUENTA="$CUENTA_AWS"

cat <<RESUMEN

===========================================================================
  Instancia:     $ID
  Elastic IP:    $IP   (fija: no cambia al apagar y prender)
  Cuenta:        $CUENTA
  ARN instancia: arn:aws:ec2:${REGION}:${CUENTA}:instance/${ID}

  MLflow:        http://${IP}:5000
  SSH:           ssh -i ~/.ssh/${NOMBRE}.pem ubuntu@${IP}

  Los clientes apuntan con:
      export MLFLOW_TRACKING_URI=http://${IP}:5000

  MLflow tarda 2-3 minutos en quedar arriba (instalacion inicial).
  Seguimiento:  ssh ... 'tail -f /var/log/user-data.log'

  Siguiente:
    1. Guardar INSTANCIA, IP_MLFLOW y SECURITY_GROUP en infra/valores.env
    2. Crear el presupuesto:  bash infra/presupuesto.sh
    3. Crear los usuarios:    bash infra/usuarios-iam.sh
===========================================================================
RESUMEN
