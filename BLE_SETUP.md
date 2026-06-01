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

## Run

Start the server as usual:

```bash
python3 web_server.py
```

BLE provisioning starts automatically in a background thread.

You can also run provisioning alone (no camera) for testing:

```bash
python3 -c "from ble_wifi_provision import run_ble_provisioning_server; run_ble_provisioning_server()"
```

## Troubleshooting

- **No adapter**: `hciconfig` or `bluetoothctl show` — enable Bluetooth in `raspi-config`.
- **Permission errors**: run with a user in the `bluetooth` group, or test with `sudo` once.
- **`characteristic_value` / notify errors**: update `ble_wifi_provision.py` to the latest version (uses `set_value()`).
- **Scan empty**: move the Pi closer to routers; run `nmcli dev wifi list` manually.
- **Connect fails**: check password; open networks leave password empty in the app.
