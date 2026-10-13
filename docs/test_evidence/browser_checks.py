"""Browser click-through checks for the ProxyLens dashboard.

    python docs/test_evidence/browser_checks.py                  # all parts
    python docs/test_evidence/browser_checks.py --parts sample   # one part

The script that produced browser_checks.txt (11 October 2026). It drives
headless Google Chrome over the Chrome DevTools protocol against a running
dashboard: it opens pages, clicks buttons, chooses options, uploads a file,
downloads files, and compares the page text and the downloaded files with
expected values. "Sample audit values match results/" reads the expected
numbers straight from the CSV files in results/ with pandas, not through the
dashboard's code. See README.md in this folder for how to re-run it.

Parts, run in this order by default:
    sample      sample audit: start screen, stored values, report, old addresses
    upload      CSV upload with a lender score (needs make_german_upload.py)
    german      built-in German Credit: guarded rollback, full repair
    homecredit  built-in Home Credit, 5,000 rows: full and guarded repair, lookup
    uci         built-in UCI Default of Credit Card Clients, 5,000 rows

Everything a run writes goes into --out (a new timestamped folder by default):
browser_checks.log (one PASS/FAIL line per check, as in browser_checks.txt),
browser_checks.jsonl (the same as JSON), screenshots/ and downloads/.
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import websocket  # pip install websocket-client

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
PARTS = ["sample", "upload", "german", "homecredit", "uci"]
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome",
]


# ==========================================================================
# a minimal headless-Chrome driver for Streamlit pages
# ==========================================================================
def find_chrome(explicit: str | None) -> str:
    for c in [explicit, os.environ.get("CHROME_PATH"), *CHROME_CANDIDATES]:
        if c and (os.path.exists(c) or shutil.which(c)):
            return c if os.path.exists(c) else shutil.which(c)
    raise SystemExit("Google Chrome not found: pass --chrome PATH or set CHROME_PATH.")


class Chrome:
    def __init__(self, chrome: str, port: int = 9333, width: int = 1440,
                 height: int = 900, download_dir: str | None = None):
        self.width, self.height = width, height
        self.profile = tempfile.mkdtemp(prefix="cdp-")  # throwaway Chrome profile
        self.proc = subprocess.Popen(
            [chrome, "--headless=new", f"--remote-debugging-port={port}",
             "--remote-allow-origins=*", f"--window-size={width},{height}",
             "--hide-scrollbars", f"--user-data-dir={self.profile}",
             "--no-first-run", "--force-device-scale-factor=1", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tabs = ver = None
        for _ in range(150):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                ver = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version"))
                break
            except Exception:
                time.sleep(0.2)
        if tabs is None:
            self.close()
            raise SystemExit(f"Chrome did not open its debugging port {port}; is another "
                             "instance using it? Try --port with a different number.")
        page = next(t for t in tabs if t["type"] == "page")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=900)
        self.id = 0
        for m in ("Page.enable", "Runtime.enable", "DOM.enable"):
            self.send(m)
        self.metrics(height)
        if download_dir:
            bws = websocket.create_connection(ver["webSocketDebuggerUrl"], timeout=60)
            bws.send(json.dumps({"id": 1, "method": "Browser.setDownloadBehavior",
                                 "params": {"behavior": "allow", "downloadPath": download_dir}}))
            bws.recv()
            self.bws = bws

    def close(self):
        """Stop Chrome and delete its profile (each run's profile is 50-150 MB)."""
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        shutil.rmtree(self.profile, ignore_errors=True)

    def send(self, method, **params):
        self.id += 1
        self.ws.send(json.dumps({"id": self.id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def metrics(self, height):
        self.send("Emulation.setDeviceMetricsOverride", width=self.width, height=int(height),
                  deviceScaleFactor=1, mobile=False)

    def js(self, expr):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:800])
        return r["result"].get("value")

    # ---- waiting -------------------------------------------------------
    def running(self):
        return self.js("""(() => {
            const w = document.querySelector('[data-testid="stStatusWidget"]');
            return !!(w && /running|connecting/i.test(w.innerText));})()""")

    def idle(self, timeout=900, settle=1.5):
        """Wait until the script has stopped running and the text is stable."""
        t0, last, since = time.time(), None, time.time()
        time.sleep(0.6)
        while time.time() - t0 < timeout:
            txt = self.text()
            if txt != last or self.running():
                last, since = txt, time.time()
            elif time.time() - since >= settle:
                return txt
            time.sleep(0.3)
        raise TimeoutError("page did not settle")

    def wait_text(self, s, timeout=900):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if s in self.text():
                return
            time.sleep(0.4)
        raise TimeoutError(f"text not found: {s}")

    def goto(self, url):
        self.send("Page.navigate", url=url)
        time.sleep(1.0)
        return self.idle()

    def text(self):
        return self.js("document.body.innerText") or ""

    # ---- input ---------------------------------------------------------
    def _box(self, expr):
        r = self.js(f"""(() => {{ const e = {expr}; if (!e) return null;
            e.scrollIntoView({{block: 'center'}}); const b = e.getBoundingClientRect();
            return [b.x + b.width / 2, b.y + b.height / 2]; }})()""")
        if r is None:
            raise LookupError(expr)
        return r

    def click_xy(self, x, y):
        for t in ("mousePressed", "mouseReleased"):
            self.send("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)

    def click(self, expr):
        self.click_xy(*self._box(expr))

    def button(self, label, settle=True):
        self.click(f"""[...document.querySelectorAll('button')].find(b =>
            [...b.querySelectorAll('p')].some(p => p.innerText.trim() === {json.dumps(label)}))""")
        if settle:
            return self.idle()

    def button_disabled(self, label):
        return self.js(f"""(() => {{ const b = [...document.querySelectorAll('button')].find(b =>
            [...b.querySelectorAll('p')].some(p => p.innerText.trim() === {json.dumps(label)}));
            return b ? b.disabled : null; }})()""")

    def tab(self, name):
        self.click(f"""[...document.querySelectorAll('[role=tab]')].find(t =>
            t.innerText.trim() === {json.dumps(name)})""")
        return self.idle()

    def _widget(self, testid, label):
        return (f"""[...document.querySelectorAll('[data-testid="{testid}"]')].find(w =>
            (w.querySelector('label') || w).innerText.split('\\n')[0].trim() === {json.dumps(label)})""")

    def key(self, k, code, vk):
        self.send("Input.dispatchKeyEvent", type="keyDown", key=k, code=code,
                  windowsVirtualKeyCode=vk)
        self.send("Input.dispatchKeyEvent", type="keyUp", key=k, code=code,
                  windowsVirtualKeyCode=vk)

    def select(self, label, option, testid="stSelectbox"):
        self.click(f"({self._widget(testid, label)}).querySelector('input')")
        time.sleep(0.4)
        self.send("Input.insertText", text=option)
        time.sleep(0.5)
        self.key("Enter", "Enter", 13)
        time.sleep(0.3)
        self.key("Escape", "Escape", 27)
        return self.idle()

    def segmented(self, label, option):
        self.click(f"""[...({self._widget('stButtonGroup', label)}).querySelectorAll('button')]
            .find(b => b.innerText.trim() === {json.dumps(option)})""")
        return self.idle()

    def radio(self, label, option):
        scope = (f"[({self._widget('stRadio', label)})]" if label else
                 "[...document.querySelectorAll('[data-testid=\"stRadio\"]')]")
        self.click(f"""{scope}.flatMap(w => [...w.querySelectorAll('label')])
            .find(l => l.innerText.trim().startsWith({json.dumps(option)}))""")
        return self.idle()

    def upload(self, path):
        doc = self.send("DOM.getDocument", depth=-1, pierce=True)
        node = self.send("DOM.querySelector", nodeId=doc["root"]["nodeId"],
                         selector='input[type="file"]')
        self.send("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(path)])
        return self.idle()

    # ---- capture -------------------------------------------------------
    def shot(self, path, full=True, top=True):
        """A PNG of the page; full=True grows the viewport to the page's height first."""
        if top:
            self.js("""document.querySelectorAll('*').forEach(e => {
                if (e.scrollTop > 0) e.scrollTop = 0; }); window.scrollTo(0, 0)""")
        if full:
            h = self.js("""Math.max(...[...document.querySelectorAll(
                '[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main, .main')]
                .map(e => e.scrollHeight), document.body.scrollHeight)""")
            self.metrics(max(self.height, min(int(h) + 20, 16000)))
            time.sleep(1.2)
        data = self.send("Page.captureScreenshot", format="png")["data"]
        Path(path).write_bytes(base64.b64decode(data))
        if full:
            self.metrics(self.height)
            time.sleep(0.4)
        return path


# ==========================================================================
# checks
# ==========================================================================
class Run:
    """Where a run reads from and writes to."""

    def __init__(self, args):
        self.base = args.base_url.rstrip("/")
        self.out = Path(args.out)
        self.shots = Path(args.screenshots) if args.screenshots else self.out / "screenshots"
        self.dl = self.out / "downloads"
        self.upload = Path(args.upload)
        for d in (self.out, self.shots, self.dl):
            d.mkdir(parents=True, exist_ok=True)
        self.log = self.out / "browser_checks.log"
        self.jsonl = self.out / "browser_checks.jsonl"
        self.results = []

    def record(self, check, ok, evidence):
        line = ("PASS " if ok else "FAIL ") + check + " | " + str(evidence)[:300]
        print(line, flush=True)
        self.results.append(bool(ok))
        with open(self.log, "a") as f:
            f.write(line + "\n")
        with open(self.jsonl, "a") as f:
            f.write(json.dumps({"check": check, "pass": bool(ok), "evidence": evidence}) + "\n")

    def note(self, text):
        print(text, flush=True)
        with open(self.log, "a") as f:
            f.write(text + "\n")

    def shot(self, c, name, **kw):
        return c.shot(self.shots / name, **kw)

    def _files(self):
        return {p: os.path.getmtime(p) for p in glob.glob(str(self.dl / "*"))
                if not p.endswith(".crdownload")}

    def download(self, c, label, timeout=60):
        """Click a download button and return the file it saved (new or overwritten)."""
        before = self._files()
        c.button(label, settle=False)
        t0 = time.time()
        while time.time() - t0 < timeout:
            new = [p for p, m in self._files().items() if before.get(p) != m]
            if new:
                time.sleep(0.5)
                c.idle()
                return new[0]
            time.sleep(0.3)
        c.idle()
        return None


def norm(t):
    return re.sub(r"\s+", " ", t or "")


def has(txt, *parts):
    return all(norm(p) in norm(txt) for p in parts)


def expected(attr, mode):
    """The values the sample audit should show, read straight from results/."""
    lt = pd.read_csv(RESULTS / "leakage_table.csv")
    lt = lt[(lt.attribute == attr) & (lt.probe == "gbm")].set_index("condition")
    ft = pd.read_csv(RESULTS / "iteration3" / "final_table.csv")
    ft = ft[ft.attribute == attr].set_index("method")
    label = {"full": "proxy_aware full (iteration 1)",
             "guarded": "proxy_aware guarded (iteration 2)"}[mode]
    q = pd.read_csv(RESULTS / "mitigation_quadrant_summary.csv")
    q = q[q.attribute == attr].set_index("arm")
    e = {"probe AUC": f"{lt.loc['actual', 'auc_mean']:.3f}",
         "probe sd": f"± {lt.loc['actual', 'auc_std']:.3f} across folds",
         "control": f"{lt.loc['shuffled', 'auc_mean']:.3f}",
         "balanced acc": f"{lt.loc['actual', 'balanced_acc_mean']:.3f}",
         "features": f"{int(lt.loc['actual', 'n_features'])}",
         "rows": f"{int(lt.loc['actual', 'n_samples']):,}",
         "danger before": f"{int(q.loc['baseline', 'n_leaky_and_relied_on'])} features both reveal"}
    for m in ["AUC", "DP diff", "DI ratio", "EO diff", "probe AUC"]:
        b, a = ft.loc["none", m], ft.loc[label, m]
        e[f"{m} after"] = f"{m}\n\n{a:.4f}\n\n{a - b:+.4f}"
    if mode == "full":
        sp = pd.read_csv(RESULTS / "iteration4" / "strong_probe.csv")
        sp = sp[sp.attribute == attr].set_index("features")
        o, r = sp.loc["original"], sp.loc["repaired, iteration 1"]
        e["weak probe"] = f"{r.weak_probe:.3f}\n\n{r.weak_probe - o.weak_probe:+.3f} from {o.weak_probe:.3f}"
        e["strong probe"] = (f"{r.strong_probe_auc:.3f}\n\n{r.strong_probe_auc - o.strong_probe_auc:+.3f}"
                             f" from {o.strong_probe_auc:.3f}")
        e["repaired"] = f"Features repaired\n\n{int(q.loc['full', 'n_treated'])}"
        tr = pd.read_csv(RESULTS / "mitigation_transitions.csv")
        tr = tr[(tr.attribute == attr) & (tr.arm == "full")]
    else:
        t = pd.read_csv(RESULTS / "iteration2" / f"test_results_{attr}.csv").set_index("candidate")
        e["repaired"] = f"Features repaired\n\n{int(t.loc['C1_guard99', 'n_treated'])}"
        e["skipped"] = f"Skipped by the guard\n\n{int(t.loc['C1_guard99', 'n_skipped'])}"
        tr = pd.read_csv(RESULTS / "iteration2" / f"transitions_{attr}.csv")
    e["manufactured"] = f"{int(tr.leakage_emerged.astype(bool).sum())} features had no detectable leakage"
    return e


def part_sample(c, R):
    c.goto(R.base + "/")
    R.shot(c, "01_start.png")
    R.record("Start screen offers both choices", "Audit my data" in c.text()
             and "Explore a sample audit" in c.text(), "01_start.png")

    for attr in ("CODE_GENDER", "REGION_RATING_CLIENT"):
        for mode in ("full", "guarded"):
            url = f"{R.base}/results?demo=1&attr={attr}&mode={mode}"
            txt = "\n".join(c.goto(f"{url}&tab={t}") for t in
                            ("Overview", "Leakage", "Explanation", "Repair"))
            exp = expected(attr, mode)
            missing = {k: v for k, v in exp.items() if not has(txt, v)}
            R.record(f"Sample audit values match results/ ({attr}, {mode})", not missing,
                     f"{len(exp) - len(missing)}/{len(exp)} stored values found"
                     + (f"; missing {missing}" if missing else ""))

    url = f"{R.base}/results?demo=1&mode=full"
    c.goto(url)
    R.shot(c, "04_overview.png")
    c.goto(url + "&tab=Leakage")
    R.shot(c, "05_leakage.png")
    c.goto(url + "&tab=Explanation")
    R.shot(c, "06_explanation.png")
    txt = c.goto(url + "&tab=Repair")
    R.shot(c, "07_repair.png")
    R.record("07_repair shows weak 0.771→0.555 and strong 0.875→0.795 (gender, full)",
             has(txt, "0.555 -0.216 from 0.771", "0.795 -0.080 from 0.875"),
             "07_repair.png")
    txt = c.goto(url + "&tab=Applicant+lookup")
    R.shot(c, "09_applicant_lookup.png")
    R.record("Applicant lookup (sample)", "Before repair" in txt and "Risk score" in txt
             and "After repair" in txt, "stored example applicants shown")
    c.goto(url + "&tab=Report")
    R.shot(c, "10_report.png")
    path = R.download(c, "Download report (Markdown)")
    body = open(path).read() if path else ""
    R.record("Report downloads (sample audit)", "## 3. Mitigation (full)" in body,
             f"{os.path.basename(path or '')}, {len(body):,} chars")

    c.goto(R.base + "/mitigate?demo=1")
    href = c.js("location.href")
    R.record("Old address /mitigate?demo=1 redirects to the Repair tab",
             "/results" in href and "tab=Repair" in href, href)
    c.goto(R.base + "/?demo=1")
    href = c.js("location.href")
    R.record("?demo=1 opens the sample audit", "/results?demo=1" in href
             and "Verdict" in c.text(), href)


def run_builtin(c, R, dataset, label, shot_running=None):
    c.goto(R.base + "/setup")
    if dataset != "Home Credit (sample)":
        c.select("Dataset", dataset)
    c.idle()
    t0 = time.time()
    c.button("Run audit", settle=False)
    if shot_running:
        c.wait_text("Step 2 of 3", 120)
        time.sleep(3)
        R.shot(c, shot_running, full=False)
    c.wait_text("Verdict", 900)
    txt = c.idle()
    secs = time.time() - t0
    R.record(f"Load and audit + explanation run ({label})",
             has(txt.upper(), "LEAKAGE", "MODEL RELIANCE", "WHAT THIS MEANS"),
             f"audit + explanation in {secs:.0f} s; "
             + re.sub(r"\s+", " ", txt[txt.find("DATASET"):txt.find("STATUS")])[:160])
    return txt


def apply(c, mode):
    """Choose a repair mode on the Repair tab and apply it; returns (page text, seconds)."""
    c.tab("Repair")
    c.radio(None, "Guarded" if mode == "guarded" else "Full")
    labels = [lab for lab in ("Apply repair", "Apply guarded repair instead",
                              "Apply full repair instead")
              if c.button_disabled(lab) is False]
    t0 = time.time()
    c.button(labels[0], settle=False)
    c.wait_text(f"Before and after · {mode} mode", 900)
    txt = c.idle()
    return txt, time.time() - t0


def part_homecredit(c, R):
    run_builtin(c, R, "Home Credit (sample)", "Home Credit, 5,000 rows",
                shot_running="03_running.png")
    txt = c.text()
    R.record("Built-in sample loads at 5,000 rows with 116 features",
             has(txt, "ROWS 5,000", "FEATURES 116"), "summary bar")
    for mode in ("full", "guarded"):
        txt, secs = apply(c, mode)
        R.record(f"Repair {mode} (Home Credit, live)", "Leakage measured two ways" in txt
                 and "Manufactured proxies" in txt, f"{secs:.0f} s in the dashboard")
    txt = c.tab("Applicant lookup")
    m = re.search(r"\d+ of [\d,]+ held-out applicants[^\n]*", txt)
    R.record("Applicant lookup (live)", "Before repair" in txt and m is not None,
             m.group(0) if m else norm(txt)[:300])
    c.tab("Report")
    path = R.download(c, "Download report (Markdown)")
    body = open(path).read() if path else ""
    R.record("Report downloads (live)", "## 3. Mitigation (guarded)" in body
             and "## 1. Leakage audit" in body, os.path.basename(path or ""))


def part_german(c, R):
    run_builtin(c, R, "German Credit", "German Credit")
    txt, secs = apply(c, "guarded")
    ok = "No feature was repaired" in txt and "rolled back" in txt
    R.record("Guarded mode undoes the round on German Credit and explains it", ok,
             re.search(r"No feature was repaired[^\n]*", txt).group(0) if ok else norm(txt)[:300])
    for _ in range(30):  # let the "repair applied" toast fade before the screenshot
        if not c.js("!!document.querySelector('[data-testid=\"stToast\"]')"):
            break
        time.sleep(0.5)
    R.shot(c, "12_guarded_undo.png")
    txt, secs = apply(c, "full")
    R.record("Repair full (German Credit)", "Features repaired" in txt, f"{secs:.0f} s")


def part_uci(c, R):
    run_builtin(c, R, "UCI Default of Credit Card Clients", "UCI Default, 5,000 rows")
    txt, secs = apply(c, "full")
    R.record("Repair full (UCI Default)", "Leakage measured two ways" in txt, f"{secs:.0f} s")


def part_upload(c, R):
    # The column names below (outcome, sex, attr_1, attr_2, attr_5, age, lender_score)
    # are those of the file make_german_upload.py writes.
    if not R.upload.exists():
        raise FileNotFoundError(f"{R.upload} not found: run make_german_upload.py first")
    c.goto(R.base + "/setup")
    c.segmented("Data source", "Upload a CSV")
    c.upload(R.upload.resolve())
    c.wait_text(f"Using {R.upload.name}")

    # invalid choices are blocked
    for widget, value, expect, what in (
            ("Outcome column", "attr_5", "looks like a measurement", "continuous outcome"),
            ("Protected attribute", "attr_2", "has 33 different values", "attribute with 33 groups")):
        txt = c.select(widget, value)
        blocked = c.button_disabled("Run audit")
        R.record(f"Setup checklist blocks invalid choice: {what}", has(txt, expect) and blocked,
                 f"Run audit disabled={blocked}")
        if what == "continuous outcome":
            R.shot(c, "11_checklist_blocked.png")
        c.select(widget, "outcome" if widget == "Outcome column" else "sex")
    c.select("Your model's score or decision (optional)", "attr_1")
    txt = c.text()
    blocked = c.button_disabled("Run audit")
    R.record("Setup checklist blocks invalid choice: non-numeric score column",
             "is not numeric" in txt and blocked, f"Run audit disabled={blocked}")

    c.select("Your model's score or decision (optional)", "lender_score")
    c.select("Columns to leave out", "age", testid="stMultiSelect")
    txt = c.idle()
    ok = (not c.button_disabled("Run audit")
          and has(txt, "18 columns used as credit features (1 left out)"))
    R.record("Valid upload passes the checklist", ok,
             "18 features after removing sex, age, lender_score")
    R.shot(c, "02_setup.png")

    c.button("Run audit", settle=False)
    c.wait_text("Verdict", 900)
    txt = c.idle()
    R.record("Upload a CSV: attribute, lender column and left-out columns removed",
             has(txt, "FEATURES 18"), "summary bar shows 18 features")
    ok = all(s in txt for s in ("YOUR MODEL'S DECISIONS", "DP diff", "DI ratio", "EO diff",
                                "Approval rate"))
    R.record("Lender score column shows DP/DI/EO", ok,
             re.search(r"DI ratio [\d.]+ \(pass[^\n]*", txt).group(0) if ok else norm(txt)[:300])

    apply(c, "full")
    c.tab("Overview")
    R.shot(c, "04b_overview_lender.png")
    c.tab("Repair")
    c.button("Prepare repaired data", settle=False)
    c.wait_text("Download repaired data (.zip)", 300)
    c.idle()
    R.shot(c, "08_repaired_data_export.png")
    path = R.download(c, "Download repaired data (.zip)")
    ok, ev = False, "no file"
    if path:
        z = zipfile.ZipFile(path)
        d = pd.read_csv(io.BytesIO(z.read("repaired_data.csv")))
        bad = {"sex", "age", "lender_score", "outcome"} & set(d.columns)
        ok = not bad and "outcome_default" in d.columns and "README.txt" in z.namelist()
        ev = (f"{os.path.basename(path)}: {d.shape[0]} rows × {d.shape[1]} columns; "
              f"removed columns present: {sorted(bad) or 'none'}")
    R.record("Repaired-data export", ok, ev)

    txt, secs = apply(c, "guarded")
    ok = "No feature was repaired" in txt and "rolled back" in txt
    R.record("Guarded mode undoes the round on the German upload and explains it", ok,
             f"{secs:.0f} s")
    txt = c.tab("Applicant lookup")
    R.record("Applicant lookup (live, upload)", "Before repair" in txt, "guarded mode")
    c.tab("Report")
    path = R.download(c, "Download report (Markdown)")
    body = open(path).read() if path else ""
    R.record("Report includes the lender's decisions", "## Your model's decisions" in body,
             os.path.basename(path or ""))


# ==========================================================================
def main():
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8502",
                    help="where the dashboard is running (default: %(default)s)")
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS,
                    help="which parts to run, in order (default: all)")
    ap.add_argument("--upload", default=str(ROOT / "data" / "german_with_lender_score.csv"),
                    help="the CSV for the upload part (default: data/german_with_lender_score.csv,"
                         " written by make_german_upload.py)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "test_evidence" / f"run_{stamp}"),
                    help="folder for the log, downloads and screenshots "
                         "(default: docs/test_evidence/run_<date_time>)")
    ap.add_argument("--screenshots", default=None,
                    help="folder for screenshots (default: <out>/screenshots); pass "
                         "docs/screenshots only to regenerate the committed ones")
    ap.add_argument("--chrome", default=None,
                    help="path to Google Chrome or Chromium (default: CHROME_PATH or a "
                         "standard install location)")
    ap.add_argument("--port", type=int, default=9333,
                    help="Chrome remote-debugging port (default: %(default)s)")
    args = ap.parse_args()

    R = Run(args)
    R.note(f"ProxyLens browser checks, run {datetime.now().isoformat(timespec='seconds')} "
           f"against {R.base}")
    c = Chrome(find_chrome(args.chrome), port=args.port, download_dir=str(R.dl))
    try:
        for part in args.parts:
            R.note(f"== {part}")
            t0 = time.time()
            try:
                globals()[f"part_{part}"](c, R)
            except Exception as e:
                R.record(f"{part}: driver error", False, repr(e)[:500])
            R.note(f"== {part} done in {time.time() - t0:.0f} s")
    finally:
        c.close()
    passed = sum(R.results)
    R.note(f"{passed} of {len(R.results)} checks passed. Output in {R.out}")
    sys.exit(0 if passed == len(R.results) else 1)


if __name__ == "__main__":
    main()
