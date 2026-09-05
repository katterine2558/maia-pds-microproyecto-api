# MLflow en AWS EC2

Servidor de seguimiento de experimentos del micro-proyecto. Una sola maquina,
compartida por el equipo: todos los experimentos caen en el mismo lugar para
poder compararlos, que es lo que pide la Entrega 2.

## Datos de la maquina

Los valores concretos (ids de cuenta e instancia, IP publica, IPs del equipo)
viven en `infra/valores.env`, **fuera del control de versiones**: este
repositorio es publico. La plantilla versionada es `infra/valores.env.example`.


| | |
|---|---|
| Instancia | `<INSTANCIA>` |
| Elastic IP | `<IP_MLFLOW>` |
| Security group | `<SECURITY_GROUP>` |
| Region | `us-east-1` |
| Tipo | `t3.micro` |
| MLflow | `http://<IP_MLFLOW>:5000` |

La Elastic IP es fija: no cambia aunque la maquina se apague y se vuelva a
prender. La URL de arriba sirve siempre.

## Montaje inicial

Se hace una sola vez, desde la cuenta duena.

```bash
# 1. Cada integrante manda su IP publica
curl ifconfig.me

# 2. Pegarlas en IPS_EQUIPO dentro de infra/ec2-mlflow.sh
# 3. Crear la maquina
bash infra/ec2-mlflow.sh

# 4. Crear la alarma de costo (USD 5/mes sobre EC2 + Elastic IP + disco)
bash infra/presupuesto.sh tu-correo@ejemplo.com
```

MLflow tarda 2-3 minutos en quedar arriba mientras se instala. Para seguir el
avance:

```bash
ssh -i ~/.ssh/mlflow-maia.pem ubuntu@<IP_MLFLOW> 'tail -f /var/log/user-data.log'
```

## Permisos del equipo

Camilo, Rainer y Leonardo tienen cada uno un **usuario IAM dentro de la
cuenta duena**. No usan su
propia cuenta de AWS: entran por la URL de esta cuenta.

```
https://<CUENTA_AWS>.signin.aws.amazon.com/console
```

Solo pueden **prender** la maquina (`infra/iam-equipo.json`). Apagarla no hace
falta: la maquina se apaga sola. Eso evita que alguien la apague en medio del
entrenamiento de otro.

Creacion de los usuarios:

```bash
# La politica ya trae la cuenta y el id de instancia reales
aws iam create-policy --policy-name prender-mlflow \
    --policy-document file://infra/iam-equipo.json

aws iam create-group --group-name equipo-mlflow
aws iam attach-group-policy --group-name equipo-mlflow \
    --policy-arn arn:aws:iam::<CUENTA_AWS>:policy/prender-mlflow

for u in camilo rainer leonardo; do
    aws iam create-user --user-name "$u"
    aws iam add-user-to-group --user-name "$u" --group-name equipo-mlflow
    aws iam create-login-profile --user-name "$u" \
        --password 'CAMBIAR-EN-EL-PRIMER-INGRESO' --password-reset-required
done
```

Con una politica tan estrecha, la consola de EC2 les va a mostrar avisos de
"no autorizado" en los paneles de volumenes y snapshots. Es normal: el boton
de Start funciona igual.

## Uso diario

**Prender la maquina** (cualquiera del equipo): consola AWS -> EC2 ->
Instances -> `mlflow-maia` -> Instance state -> Start instance. Arranca en
~40 segundos.

**Apuntar los scripts al servidor**, antes de entrenar:

```bash
export MLFLOW_TRACKING_URI=http://<IP_MLFLOW>:5000
```

O dentro del codigo:

```python
import mlflow

mlflow.set_tracking_uri("http://<IP_MLFLOW>:5000")
mlflow.set_experiment("readmision-diabetes")
```

**Ver los experimentos**: `http://<IP_MLFLOW>:5000` en el navegador.

## Apagado automatico

La maquina se apaga sola para no gastar de mas. Tres reglas, en orden:

1. **Tope duro**: 12 horas encendida, se apaga pase lo que pase.
2. **Sesion SSH abierta**: cuenta como actividad, no se apaga.
3. **Sin peticiones a MLflow por 3 horas**: se apaga.

El tope duro existe porque la UI de MLflow refresca sola: una pestana olvidada
mantendria la maquina viva indefinidamente.

Si alguien va a correr algo largo y no quiere que se apague:

```bash
ssh -i ~/.ssh/mlflow-maia.pem ubuntu@<IP_MLFLOW>
mantener-prendida     # desactiva el apagado
reanudar-apagado      # lo vuelve a activar -- no se te olvide
```

Los umbrales estan en `/usr/local/bin/apagar-si-inactivo`.

## Costo

| Concepto | Encendida | Apagada |
|---|---|---|
| `t3.micro` | 0,25/dia | 0 |
| Elastic IP | 0,12/dia | 0,12/dia |
| Disco 10 GB | 0,03/dia | 0,03/dia |

Con el apagado automatico, el proyecto completo cuesta ~USD 3.

La alarma de presupuesto `maia-pds-ec2` avisa a <CORREO_ALERTAS> al
60%, al 80% y cuando la proyeccion del mes pase los USD 5.

La cuenta ya tenia otro presupuesto, `maia-pds-alerta-1usd`: USD 1 **sobre
todos los servicios**, sin filtro. Ese se va a disparar apenas la EC2 lleve
unos dias corriendo. No es un error; conviene subirlo o borrarlo para que no
tape los avisos que si importan.

## Pantallazos para la entrega

El enunciado exige que se vea **el usuario y la IP de la maquina en EC2, y la
IP en MLflow**. Tres capturas:

1. **Terminal SSH conectada**, donde se lea el prompt `ubuntu@ip-172-31-x-x`.
2. **Consola de EC2** con la instancia y su IP publica.
3. **UI de MLflow** con los experimentos, donde se lea `http://<IP_MLFLOW>:5000`
   en la barra de direcciones.

Van a `docs/entregas/figuras/`.

Como respaldo, exportar las metricas de los runs a `docs/soportes/mlflow/`.
Si la maquina se pierde, ese CSV es la evidencia que sobrevive.

## Al terminar

**Hasta que califiquen la Entrega 3, la maquina se deja detenida, nunca
terminada.** El enunciado lo pide asi para poder verificarla.

Despues de la nota:

```bash
aws ec2 terminate-instances --instance-ids <INSTANCIA> --region us-east-1
aws ec2 release-address --allocation-id ALLOCATION_ID --region us-east-1
for u in camilo rainer leonardo; do aws iam delete-user --user-name "$u"; done
```

Liberar la Elastic IP es un paso aparte: una direccion reservada sin usar sigue
facturando aunque la instancia ya no exista.

## Seguridad

- Los puertos 22 y 5000 estan abiertos **solo a las IPs del equipo**. Si a
  alguien le cambia la IP de su casa, hay que agregarla al security group.
- MLflow **no trae autenticacion**. Lo unico que lo protege es el security
  group. Nunca abrirlo a `0.0.0.0/0`.
- La llave `~/.ssh/mlflow-maia.pem` no se sube al repo ni se comparte por chat.
  Para el uso normal (ver MLflow, registrar runs) no hace falta SSH.
- Usar un usuario IAM administrador para el dia a dia, no la cuenta root.
