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

Allow specific ip address for client of the web server:

```bash
sudo ufw allow from 188.12.139.142 to any port 8000
```

Allow a range of ip addresses for **all** the ports:

```bash
sudo ufw allow from 192.168.0.0/24
```

Then disallow all the others:

```bash
sudo ufw default deny incoming
```

#### Verifica porte

Il comando `ss` può mostrare le porte che stanno ascoltando le connessioni e le reti da cui accetta tali connessioni. È un’alternativa moderna al vecchio comando netstat.
Esegui il seguente comando per visualizzare le porte aperte:

```bash
sudo ss -ltn
```

#### View UFW logs and status

```bash
tac /var/log/ufw.log | head -n 20
```

To see the status:

```bash
sudo ufw status verbose
```

With priority:

```bash
sudo ufw status numbered
```

You should see something like:

```bash
sudo ufw status verbose
Status: active
Logging: on (low)
Default: deny (incoming), allow (outgoing), deny (routed)
New profiles: skip

To                         Action      From
--                         ------      ----
Anywhere                   ALLOW IN    72.80.205.0/24            
22                         ALLOW IN    Anywhere                  
8000                       ALLOW IN    188.12.139.142            
```

### Log Rotation

To use logrotate for managing your logs, you need to create a configuration file for logrotate. Here's a step-by-step plan:

1. Create a new logrotate configuration file, e.g., `/etc/logrotate.d/uvicorn`.
In this file, specify the path to your log files, the rotation interval, the number of backups to keep, and any other options you want to use.

```txt
/home/user/projects/wl-semsearch-poc/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0644 user group
}
```

This configuration will rotate the log files daily, keep 7 days of backups, compress rotated files, and create new log files with the specified permissions and ownership if they don't exist.

Finally, ensure that `logrotate` is run daily by the cron daemon,  you can add a cron job manually with `crontab -e` and adding the following line:

```bash
crontab -e

@daily /usr/sbin/logrotate /etc/logrotate.conf
```

This will run logrotate daily using the main configuration file /etc/logrotate.conf, which includes all files in the /etc/logrotate.d directory.
