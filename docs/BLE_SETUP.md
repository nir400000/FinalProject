# Bluetooth Wi-Fi setup (Raspberry Pi)

The monitor advertises over **Bluetooth LE** as `BabyMonitor`. The Android app connects,
scans nearby Wi-Fi networks on the Pi, sends credentials, and receives the new IP address.

## One-time Pi setup

```bash
sudo apt update
sudo apt install -y python3-dbus python3-gi network-manager bluez

sudo systemctl enable --now bluetooth

pip install -r requirements-ble.txt
```

Ensure NetworkManager manages Wi-Fi (`nmcli dev status`).

### Wi-Fi permissions (required once)

Creating Wi-Fi profiles needs NetworkManager privileges. Choose **one**:

**Option A — polkit (recommended)**

```bash
sudo cp deploy/polkit/50-babymonitor-wifi.rules /etc/polkit-1/rules.d/
sudo usermod -aG netdev YOUR_USERNAME
sudo systemctl restart polkit
# log out and back in (or reboot)
```

**Option B — passwordless sudo for nmcli**

Edit `deploy/sudoers/babymonitor-nmcli` (replace `USER` with your username), then:

```bash
sudo cp deploy/sudoers/babymonitor-nmcli /etc/sudoers.d/babymonitor-nmcli
sudo chmod 440 /etc/sudoers.d/babymonitor-nmcli
```

The server automatically retries `nmcli` with `sudo -n` when it gets “insufficient privileges”.

### Headless Bluetooth pairing (no keyboard on the Pi)

The server registers a **NoInputNoOutput** BlueZ agent so the Pi auto-accepts pairing.
You should only need to tap **Pair** on the phone once — not confirm matching codes on the monitor.

Optional (recommended), add to `/etc/bluetooth/main.conf` under `[General]`:

```ini
JustWorksRepairing = always
```

Then restart Bluetooth:

```bash
sudo systemctl restart bluetooth
```

## Run

Start the server as usual:

```bash
python3 run.py
```

(`python3 web_server.py` also works — same entry point.)

BLE provisioning starts automatically in a background thread.

You can also run provisioning alone (no camera) for testing:

```bash
python3 -c "from server.provisioning.ble_wifi_provision import run_ble_provisioning_server; run_ble_provisioning_server()"
```

## Troubleshooting

- **No adapter**: `hciconfig` or `bluetoothctl show` — enable Bluetooth in `raspi-config`.
- **Permission errors**: run with a user in the `bluetooth` group, or test with `sudo` once.
- **`characteristic_value` / notify errors**: update `server/provisioning/ble_wifi_provision.py` to the latest version (uses `set_value()`).
- **Scan empty / missing networks**: move the Pi closer to routers; scan waits 3s after rescan and shows up to 50 networks. Compare with `nmcli dev wifi list ifname wlan0`.
- **Connect fails**: check password; open networks leave password empty in the app.
- **`insufficient privileges`**: install polkit rule or sudoers file above; add user to `netdev` group.
- **`key-mgmt: property is missing`**: update `server/provisioning/wifi_manager.py` (sets `802-11-wireless-security.key-mgmt` on a connection profile).
- **`invalid extra argument, wifi-sec.key-mgmt`**: update `wifi_manager.py` — do not pass security flags to `nmcli dev wifi connect`.
- **Pairing asks for matching codes on Pi and phone**: copy provisioning modules and restart the server. Run as a user in the `bluetooth` group or test with `sudo` once.
- **Wi-Fi list JSON parse error / 512 chars**: update provisioning modules and rebuild the app (chunked BLE transfer).
