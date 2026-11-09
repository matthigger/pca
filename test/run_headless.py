"""Drive the site in headless Chromium over CDP: run JS, take screenshots.

Usage (unsandboxed; snap Chromium cannot read hidden dirs):
    ~/venv_ds4400/bin/python test/run_headless.py URL [--wait JS]
        [--eval JS]... [--shot PATH]... [--width W] [--height H]

--wait polls a JS expression until it is truthy (60 s cap). Each --eval
runs in order (promises awaited) and prints its value; each --shot saves a
full-page PNG after the evals. Use the engine check with
    test/run_headless.py file://$PWD/test/index.html \\
        --wait '/PASS|FAIL/.test(document.title)' --eval 'out.textContent'
"""
import argparse
import base64
import json
import pathlib
import subprocess
import time
import urllib.request

import websocket

CHROME = "/snap/chromium/current/usr/lib/chromium-browser/chrome"
PROFILE = pathlib.Path.home() / "snap/chromium/common/pca_headless"


def main(a: argparse.Namespace) -> None:
    """Launch Chromium, load a.url, run the waits / evals / shots."""
    PROFILE.mkdir(parents=True, exist_ok=True)
    port = 9333
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--remote-debugging-port={port}", f"--user-data-dir={PROFILE}",
         f"--window-size={a.width},{a.height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/list"))
                tab = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        ws = websocket.create_connection(tab["webSocketDebuggerUrl"],
                                         suppress_origin=True)
        n = [0]

        def call(method: str, **params) -> dict:
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": method,
                                "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == n[0]:
                    return msg.get("result", msg)

        def ev(js: str):
            r = call("Runtime.evaluate", expression=js, returnByValue=True,
                     awaitPromise=True)
            if "exceptionDetails" in r:
                return "EXCEPTION " + json.dumps(r["exceptionDetails"])[:800]
            return r.get("result", {}).get("value")

        call("Runtime.enable")
        call("Page.navigate", url=a.url)
        time.sleep(0.5)
        if a.wait:
            t0 = time.time()
            while not ev(a.wait) and time.time() - t0 < 60:
                time.sleep(0.25)
        for js in a.eval or []:
            print(ev(js))
        for path in a.shot or []:
            h = ev("document.documentElement.scrollHeight")
            call("Emulation.setDeviceMetricsOverride", width=a.width,
                 height=int(h), deviceScaleFactor=1, mobile=False)
            time.sleep(0.3)
            r = call("Page.captureScreenshot", format="png")
            pathlib.Path(path).write_bytes(base64.b64decode(r["data"]))
            print("saved", path)
    finally:
        proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url")
    ap.add_argument("--wait")
    ap.add_argument("--eval", action="append")
    ap.add_argument("--shot", action="append")
    ap.add_argument("--width", type=int, default=1360)
    ap.add_argument("--height", type=int, default=900)
    main(ap.parse_args())
