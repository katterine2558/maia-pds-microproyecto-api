# MLflow en AWS EC2

Servidor de seguimiento de experimentos del micro-proyecto. Una sola maquina,
compartida por el equipo: todos los experimentos caen en el mismo lugar para
poder compararlos, que es lo que pide la Entrega 2.

Los valores concretos (ids de cuenta e instancia, IP publica, correo de
alertas) viven en `infra/valores.env`, **fuera del control de versiones**:
este repositorio es publico. La plantilla versionada es
`infra/valores.env.example`.

## Arquitectura

```
cliente (portatil)  --HTTP + usuario/clave-->  nginx  :5000   (publico)
                                                 |
                                                 v
                                             MLflow  :5001   (solo local)
                                                 |
                                          SQLite + artefactos en el disco EBS
```

nginx es quien pide la contrasena. MLflow escucha unicamente en `127.0.0.1`,
asi que no hay forma de llegarle sin pasar por la autenticacion.

**Por que no se usa la autenticacion propia de MLflow.** Guarda las claves con
scrypt, que reserva ~32 MB y tarda ~122 ms *por peticion*. Como la verificacion
ocurre en cada llamada a la API y un entrenamiento hace cientos, con 4 workers
en paralelo sobre 1 GB de RAM el kernel terminaba matando el proceso: se
midieron 16 muertes por OOM en 30 minutos. nginx con bcrypt de coste 5 gasta
~4 KB y **4 ms**. Mismo resultado, 30 veces mas barato.

## Datos de la maquina

| | |
|---|---|
| Instancia | `INSTANCIA` (ver `infra/valores.env`) |
| Elastic IP | `IP_MLFLOW` — fija, no cambia al apagar y prender |
| Security group | puerto 5000 abierto a internet, puerto 22 solo a la duena |
| Region | `us-east-1` |
| Tipo | `t3.micro` |
| MLflow | 3.3.2, SQLite de backend, artefactos servidos por el propio servidor |

## Montaje desde cero

En este orden:

```bash
# 1. Crear la maquina (pide las IPs del equipo solo para SSH)
bash infra/ec2-mlflow.sh

# 2. Anotar INSTANCIA, IP_MLFLOW y SECURITY_GROUP en infra/valores.env

# 3. nginx delante con usuario y contrasena
bash infra/mlflow-proxy.sh

# 4. Abrir el puerto 5000 (aborta solo si el paso 3 no quedo bien)
bash infra/abrir-puerto.sh

# 5. Alarma de costo de USD 5 sobre toda la cuenta
bash infra/presupuesto.sh

# 6. Usuarios de AWS para el equipo
bash infra/usuarios-iam.sh
```

Entre el paso 1 y el 3 hay que esperar 2-3 minutos a que termine la
instalacion inicial. Para seguirla:

```bash
ssh -i ~/.ssh/mlflow-maia.pem ubuntu@IP_MLFLOW 'tail -f /var/log/user-data.log'
```

## Uso diario

**Prender la maquina** (cualquiera del equipo, con su usuario de AWS):
consola -> EC2 -> Instances -> `mlflow-maia` -> Instance state -> Start.
Arranca en ~40 segundos y la IP no cambia.

**Registrar experimentos.** Cada integrante exporta sus credenciales antes de
entrenar:

```bash
export MLFLOW_TRACKING_URI=http://IP_MLFLOW:5000
export MLFLOW_TRACKING_USERNAME=<su-usuario>
export MLFLOW_TRACKING_PASSWORD=<su-clave>
```

O en el codigo:

```python
import mlflow

mlflow.set_tracking_uri("http://IP_MLFLOW:5000")
mlflow.set_experiment("readmision-diabetes")
```

**Ver los experimentos**: `http://IP_MLFLOW:5000` en el navegador. El
navegador pide usuario y contrasena.

Las credenciales de MLflow no estan en el repositorio. Se reparten por canal
privado y se pueden regenerar con `bash infra/mlflow-proxy.sh`.

## Permisos de AWS

Camilo, Rainer y Leonardo tienen un **usuario IAM dentro de la cuenta duena**.
No usan su propia cuenta de AWS: entran por la URL de esta cuenta.

```
https://CUENTA_AWS.signin.aws.amazon.com/console
```

Solo pueden **prender** la maquina (`infra/iam-equipo.json`). Apagarla no hace
falta: se apaga sola. Eso evita que alguien la apague en medio del
entrenamiento de otro.

Con una politica tan estrecha, la consola de EC2 les muestra avisos de "no
autorizado" en los paneles de volumenes y snapshots. Es normal: el boton de
Start funciona igual.

## Apagado automatico

La maquina se apaga sola para no gastar de mas. Tres reglas, en orden:

1. **Tope duro**: 12 horas encendida, se apaga pase lo que pase.
2. **Sesion SSH abierta**: cuenta como actividad, no se apaga.
3. **Sin peticiones por 3 horas** (log de acceso de nginx): se apaga.

El tope duro existe porque la interfaz de MLflow refresca sola: una pestana
olvidada mantendria la maquina viva indefinidamente.

Si alguien va a correr algo largo:

```bash
ssh -i ~/.ssh/mlflow-maia.pem ubuntu@IP_MLFLOW
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

El presupuesto `maia-pds-total` avisa al 60%, al 80% y cuando la proyeccion
del mes pase los USD 5 sobre el gasto **total** de la cuenta.

## Pantallazos para la entrega

El enunciado exige que se vea **el usuario y la IP de la maquina en EC2, y la
IP en MLflow**. Tres capturas:

1. **Terminal SSH conectada**, donde se lea el prompt `ubuntu@ip-172-31-x-x`.
2. **Consola de EC2** con la instancia y su IP publica.
3. **Interfaz de MLflow** con los experimentos, donde se lea
   `http://IP_MLFLOW:5000` en la barra de direcciones.

Van a `docs/entregas/figuras/`.

Como respaldo, exportar las metricas de los runs a `docs/soportes/mlflow/`.
Si la maquina se pierde, ese CSV es la evidencia que sobrevive.

## Al terminar

**Hasta que califiquen la Entrega 3, la maquina se deja detenida, nunca
terminada.** El enunciado lo pide asi para poder verificarla.

Despues de la nota:

```bash
aws ec2 terminate-instances --instance-ids INSTANCIA --region us-east-1
aws ec2 release-address --allocation-id ALLOCATION_ID --region us-east-1
for u in camilo rainer leonardo; do aws iam delete-user --user-name "$u"; done
```

Liberar la Elastic IP es un paso aparte: una direccion reservada sin usar
sigue facturando aunque la instancia ya no exista.

## Seguridad

- **El puerto 5000 esta abierto a internet.** Lo unico que protege el servidor
  es la contrasena de nginx. Que nadie reutilice ahi una clave de otro
  servicio.
- **Las credenciales viajan sobre HTTP plano**, en base64, sin cifrar. Quien
  este en la ruta de red puede leerlas. La solucion es TLS con un nombre de
  dominio; queda pendiente y no bloquea la entrega.
- El puerto 22 sigue restringido a la IP de la duena. Para entrar desde otra
  red hay que agregar esa IP al security group a mano.
- La llave `~/.ssh/mlflow-maia.pem` no se sube al repositorio ni se comparte
  por chat. Para el uso normal (ver MLflow, registrar runs) no hace falta SSH.
- Usar un usuario IAM administrador para el dia a dia, no la cuenta root.
