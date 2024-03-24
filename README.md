# Yet Another Simple Models Server

A simple server to upload data, train and serve ML models.

## Start and stop

```bash
./start_app.sh
```

```bash
./stop_app.sh
```

## API docs

```
http://localhost:8000/docs
```

### Logs
```bash
cat app.log
```

```bash
cat uvicorn.log
```

```bash
cat uvicorn.err.log
```

## ArubaCloud Setup

1. Connect to remote server
```bash
scripts/ssh-connect-aruba.sh
```

2. Clone the repo remotely

```bash
git clone...
```

3. Execute the Aruba setup script on the remote server
```bash
scripts/setup-aruba.sh
```

### UFW [UncomplicatedFirewall](https://wiki.ubuntu.com/UncomplicatedFirewall)

```bash
sudo ufw allow ssh
```

```bash
sudo ufw enable
```

```bash
sudo ufw allow 8000/tcp
```

#### Verifica porte

Il comando `ss` può mostrare le porte che stanno ascoltando le connessioni e le reti da cui accetta tali connessioni. È un’alternativa moderna al vecchio comando netstat.
Esegui il seguente comando per visualizzare le porte aperte:

```bash
sudo ss -ltn
```

#### View UFW logs

```bash
tac /var/log/ufw.log | head -n 20
```





