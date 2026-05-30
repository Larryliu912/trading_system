import os
import sys
import json
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response, stream_with_context

app = Flask(__name__)

REPO_ROOT = Path(__file__).parent.parent
REPORTS_DIR = REPO_ROOT / "single_stock_research" / "reports"

# In-memory job store (keyed by job_id)
jobs: dict = {}


def parse_report_stem(stem: str) -> dict:
    """Extract ticker, provider, short_term from a report filename stem.

    Handles two conventions:
      TICKER_YYYYMMDD_HHMMSS_PROVIDER[_short_term]   (standard)
      TICKER_YYYY-MM-DD_PROVIDER_*                    (older style)
    """
    parts = stem.split("_")
    ticker = parts[0] if parts else stem
    provider = ""
    short_term = stem.endswith("short_term")

    for i, p in enumerate(parts[1:], 1):
        if len(p) == 8 and p.isdigit():
            # standard: YYYYMMDD at index i, HHMMSS at i+1, provider at i+2
            if i + 2 < len(parts):
                provider = parts[i + 2]
            break
        # older: YYYY-MM-DD (contains dashes, length 10)
        if len(p) == 10 and p.count("-") == 2:
            if i + 1 < len(parts):
                provider = parts[i + 1]
            break

    return {"ticker": ticker, "provider": provider, "short_term": short_term}


def list_reports() -> list:
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for f in REPORTS_DIR.glob("*.md"):
        stat = f.stat()
        info = parse_report_stem(f.stem)
        reports.append({
            "filename": f.name,
            "ticker": info["ticker"],
            "provider": info["provider"],
            "short_term": info["short_term"],
            "mtime": stat.st_mtime,
            "mtime_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    reports.sort(key=lambda x: x["mtime"], reverse=True)
    return reports


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reports")
def api_reports():
    return jsonify(list_reports())


@app.route("/api/reports/<path:filename>")
def api_report(filename):
    path = (REPORTS_DIR / filename).resolve()
    try:
        path.relative_to(REPORTS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Access denied"}), 403
    if not path.exists() or not path.is_file():
        return jsonify({"error": "Not found"}), 404
    return jsonify({"content": path.read_text(encoding="utf-8", errors="replace")})


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    provider = data.get("provider", "qwen")
    short_term = bool(data.get("short_term", False))
    portfolio = data.get("portfolio", "").strip()

    if not ticker or not ticker.replace(".", "").replace("-", "").replace("^", "").isalnum():
        return jsonify({"error": "Invalid ticker"}), 400
    if provider not in ("openai", "deepseek", "qwen", "claude"):
        return jsonify({"error": "Invalid provider"}), 400

    job_id = f"{ticker}_{datetime.now().strftime('%H%M%S%f')}"
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"queue": q, "status": "running", "ticker": ticker}

    def run():
        cmd = [
            sys.executable,
            str(REPO_ROOT / "single_stock_research" / "main.py"),
            ticker,
            "--provider", provider,
        ]
        if short_term:
            cmd.append("--short-term")
        if portfolio:
            cmd.extend(["--portfolio", portfolio])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=os.environ.copy(),
            )
            for line in proc.stdout:
                q.put({"type": "log", "data": line.rstrip("\n\r")})
            proc.wait()
            if proc.returncode == 0:
                jobs[job_id]["status"] = "done"
                q.put({"type": "done", "data": "Research complete! Refresh the sidebar to see the new report."})
            else:
                jobs[job_id]["status"] = "error"
                q.put({"type": "error", "data": f"Process exited with code {proc.returncode}"})
        except Exception as e:
            jobs[job_id]["status"] = "error"
            q.put({"type": "error", "data": str(e)})
        finally:
            q.put(None)  # sentinel

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404

    def generate():
        q = jobs[job_id]["queue"]
        while True:
            try:
                item = q.get(timeout=30)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print(f"Reports directory: {REPORTS_DIR}")
    print(f"Found {len(list_reports())} existing reports")
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
