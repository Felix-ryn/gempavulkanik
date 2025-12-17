# gps_server.py

"""
Simple GPS relay:
- Run: python gps_server.py
- Open http://localhost:5000 in browser (laptop) and klik "Kirim Lokasi".
- Browser akan meminta izin lokasi; bila diizinkan, coords dikirim ke server.
- Server menyimpan ke gps_log.csv
"""

from flask import Flask, request, jsonify, render_template_string, send_file
import csv
import os
from datetime import datetime
import threading

app = Flask(__name__)

CSV_PATH = "gps_log.csv"
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["received_at_utc", "client_timestamp", "latitude", "longitude", "altitude", "accuracy", "speed", "raw_json"])

HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GPS Relay</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; max-width: 700px; }
    pre { background:#f4f4f4; padding:10px; border-radius:6px; }
    button { padding:10px 16px; font-size:16px; margin-right:8px; }
  </style>
</head>
<body>
  <h2>GPS Relay — Kirim lokasi ke server</h2>
  <p>Buka halaman ini di perangkat yang memiliki GPS (HP) untuk lokasi paling akurat.
     Jika di laptop, browser menggunakan Wi-Fi/IP positioning (kurang akurat).</p>

  <button id="sendBtn">Kirim Lokasi Sekali</button>
  <button id="watchBtn">Mulai Watch (update berkala)</button>
  <button id="stopWatchBtn" disabled>Stop Watch</button>
  <button id="downloadBtn">Download CSV dari server</button>

  <h3>Status</h3>
  <pre id="status">Ready.</pre>

<script>
const statusEl = document.getElementById('status');
let watchId = null;

function log(msg) {
  statusEl.textContent = new Date().toISOString() + " — " + msg + "\\n" + statusEl.textContent;
}

async function sendPayload(payload) {
  try {
    const res = await fetch('/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const j = await res.json();
    if (res.ok) {
      log("Server OK: " + j.message);
    } else {
      log("Server error: " + j.message);
    }
  } catch (e) {
    log("Network error: " + e.message);
  }
}

document.getElementById('sendBtn').onclick = () => {
  if (!navigator.geolocation) {
    log("Geolocation tidak didukung browser.");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const c = pos.coords;
      const payload = {
        client_timestamp: new Date(pos.timestamp).toISOString(),
        latitude: c.latitude,
        longitude: c.longitude,
        altitude: c.altitude,
        accuracy: c.accuracy,
        speed: c.speed,
        raw: pos
      };
      log("Mengirim: " + JSON.stringify({lat:c.latitude, lon:c.longitude, acc:c.accuracy}));
      sendPayload(payload);
    },
    (err) => {
      log("Error geolocation: " + err.message);
    },
    { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
  );
};

document.getElementById('watchBtn').onclick = () => {
  if (!navigator.geolocation) { log("Tidak support."); return; }
  if (watchId !== null) { log("Sudah watch."); return; }
  watchId = navigator.geolocation.watchPosition(
    (pos) => {
      const c = pos.coords;
      const payload = {
        client_timestamp: new Date(pos.timestamp).toISOString(),
        latitude: c.latitude,
        longitude: c.longitude,
        altitude: c.altitude,
        accuracy: c.accuracy,
        speed: c.speed,
        raw: pos
      };
      log("Watch update — " + c.latitude + "," + c.longitude + " acc:" + c.accuracy);
      sendPayload(payload);
    },
    (err) => { log("Watch error: " + err.message); },
    { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
  );
  document.getElementById('stopWatchBtn').disabled = false;
};

document.getElementById('stopWatchBtn').onclick = () => {
  if (watchId === null) return;
  navigator.geolocation.clearWatch(watchId);
  watchId = null;
  document.getElementById('stopWatchBtn').disabled = true;
  log("Stopped watch.");
};

document.getElementById('downloadBtn').onclick = () => {
  window.location = '/download';
};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/report", methods=["POST"])
def report():
    """
    Expect JSON:
    {
      "client_timestamp": "2025-09-30T03:00:00.000Z",
      "latitude": ...,
      "longitude": ...,
      "altitude": ...,
      "accuracy": ...,
      "speed": ...,
      "raw": { ... }   # optional
    }
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"message": f"Invalid JSON: {e}"}), 400

    lat = data.get("latitude")
    lon = data.get("longitude")
    if lat is None or lon is None:
        return jsonify({"message": "latitude/longitude missing"}), 400

    received_at = datetime.utcnow().isoformat()
    row = [
        received_at,
        data.get("client_timestamp"),
        lat,
        lon,
        data.get("altitude"),
        data.get("accuracy"),
        data.get("speed"),
        str(data.get("raw", ""))
    ]

    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        return jsonify({"message": f"Failed to write CSV: {e}"}), 500

    print(f"[{received_at}] lat={lat} lon={lon} acc={data.get('accuracy')} speed={data.get('speed')}")
    return jsonify({"message": "saved"}), 200

@app.route("/download")
def download():
    return send_file(CSV_PATH, as_attachment=True)

def run():
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    run()