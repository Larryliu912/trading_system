import os
import sys
import json
import uuid
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, Response, stream_with_context

app = Flask(__name__)

REPO_ROOT = Path(__file__).parent.parent
REPORTS_DIR        = REPO_ROOT / "single_stock_research" / "reports"
SHORT_TERM_DIR     = REPORTS_DIR / "short_term"
LONG_TERM_DIR      = REPORTS_DIR / "long_term"
HYPERSCALER_DIR    = REPORTS_DIR / "hyperscaler"

# In-memory job store (keyed by job_id)
jobs: dict = {}

# In-memory schedule store (keyed by schedule_id)
schedules: dict = {}


def _seconds_until(hhmm: str) -> float:
    now = datetime.now()
    h, m = map(int, hhmm.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _next_weekday_run_dt(hhmm: str) -> datetime:
    """Return the next Mon–Fri datetime at the given HH:MM, skipping Sat/Sun."""
    now = datetime.now()
    h, m = map(int, hhmm.split(":"))
    candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:  # 5=Sat, 6=Sun
        candidate += timedelta(days=1)
    return candidate


def _execute_schedule(schedule_id: str):
    sched = schedules.get(schedule_id)
    if not sched or sched["status"] == "cancelled":
        return
    sched["status"] = "running"
    sched["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_type = sched.get("task_type", "short_term")

    if task_type == "hyperscaler":
        try:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "single_stock_research" / "main.py"),
                "--hyperscaler",
                "--provider", sched["provider"],
            ]
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=os.environ.copy())
        except Exception:
            pass
    else:
        def _run_ticker(ticker):
            try:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "single_stock_research" / "main.py"),
                    ticker,
                    "--provider", sched["provider"],
                    "--short-term",
                ]
                subprocess.run(cmd, cwd=str(REPO_ROOT), env=os.environ.copy())
            except Exception:
                pass

        threads = [threading.Thread(target=_run_ticker, args=[t], daemon=True) for t in sched.get("tickers", [])]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    # Re-schedule: weekly keeps a fixed 7-day cadence; daily skips Sat/Sun
    if sched.get("frequency") == "weekly":
        h, m = map(int, sched["time"].split(":"))
        next_run_dt = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(days=7)
        delay = _seconds_until(sched["time"]) + 6 * 86400
    else:
        next_run_dt = _next_weekday_run_dt(sched["time"])
        delay = (next_run_dt - datetime.now()).total_seconds()
    sched["next_run"] = next_run_dt.strftime("%Y-%m-%d %H:%M")
    sched["status"] = "scheduled"

    timer = threading.Timer(delay, _execute_schedule, args=[schedule_id])
    timer.daemon = True
    timer.start()
    sched["timer"] = timer


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
    reports = []
    for subdir, report_type in (
        (SHORT_TERM_DIR,   "short_term"),
        (LONG_TERM_DIR,    "long_term"),
        (HYPERSCALER_DIR,  "hyperscaler"),
    ):
        if not subdir.exists():
            continue
        for f in subdir.glob("*.md"):
            stat = f.stat()
            info = parse_report_stem(f.stem)
            reports.append({
                "filename":    f"{subdir.name}/{f.name}",
                "ticker":      info["ticker"],
                "provider":    info["provider"],
                "short_term":  report_type == "short_term",
                "report_type": report_type,
                "mtime":       stat.st_mtime,
                "mtime_str":   datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    reports.sort(key=lambda x: x["mtime"], reverse=True)
    return reports


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reports")
def api_reports():
    return jsonify(list_reports())


@app.route("/api/reports/<path:filename>", methods=["GET", "DELETE"])
def api_report(filename):
    path = (REPORTS_DIR / filename).resolve()
    try:
        path.relative_to(REPORTS_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Access denied"}), 403
    if not path.exists() or not path.is_file():
        return jsonify({"error": "Not found"}), 404

    if request.method == "DELETE":
        path.unlink()
        return jsonify({"ok": True})

    return jsonify({
        "content":    path.read_text(encoding="utf-8", errors="replace"),
        "short_term": path.parent.name == "short_term",
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.json or {}
    provider = data.get("provider", "qwen")
    task_type = data.get("task_type", "stock")  # "stock" | "hyperscaler"

    if provider not in ("openai", "deepseek", "qwen", "claude"):
        return jsonify({"error": "Invalid provider"}), 400

    if task_type == "hyperscaler":
        job_id = f"hyperscaler_{datetime.now().strftime('%H%M%S%f')}"
        q: queue.Queue = queue.Queue()
        jobs[job_id] = {"queue": q, "status": "running", "ticker": "hyperscaler"}

        def run_hyperscaler():
            cmd = [
                sys.executable,
                str(REPO_ROOT / "single_stock_research" / "main.py"),
                "--hyperscaler",
                "--provider", provider,
            ]
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
                    q.put({"type": "done", "data": "Hyperscaler analysis complete! Refresh the sidebar to see the new report."})
                else:
                    jobs[job_id]["status"] = "error"
                    q.put({"type": "error", "data": f"Process exited with code {proc.returncode}"})
            except Exception as e:
                jobs[job_id]["status"] = "error"
                q.put({"type": "error", "data": str(e)})
            finally:
                q.put(None)

        threading.Thread(target=run_hyperscaler, daemon=True).start()
        return jsonify({"job_id": job_id})

    # Default: single-stock analysis
    ticker = data.get("ticker", "").upper().strip()
    short_term = bool(data.get("short_term", True))
    portfolio = data.get("portfolio", "").strip()

    if not ticker or not ticker.replace(".", "").replace("-", "").replace("^", "").isalnum():
        return jsonify({"error": "Invalid ticker"}), 400

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


@app.route("/api/chat/new", methods=["POST"])
def api_chat_new():
    return jsonify({"session_id": str(uuid.uuid4())})


@app.route("/api/chat/<session_id>/send", methods=["POST"])
def api_chat_send(session_id):
    data = request.json or {}
    message = (data.get("message") or "").strip()
    provider = data.get("provider", "deepseek")

    if not message:
        return jsonify({"error": "Empty message"}), 400
    if provider not in ("openai", "deepseek", "qwen", "claude"):
        return jsonify({"error": "Invalid provider"}), 400

    job_id = f"chat_{datetime.now().strftime('%H%M%S%f')}"
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"queue": q, "status": "running"}

    def run():
        cmd = [
            sys.executable,
            str(REPO_ROOT / "single_stock_research" / "main.py"),
            "--chat", message,
            "--session-id", session_id,
            "--provider", provider,
        ]
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
                line = line.rstrip("\n\r")
                if line:
                    q.put({"type": "log", "data": line})
            proc.wait()
        except Exception as e:
            q.put({"type": "error", "data": str(e)})
        finally:
            jobs[job_id]["status"] = "done"
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/chat/<session_id>/history", methods=["GET"])
def api_chat_history(session_id):
    path = REPO_ROOT / "single_stock_research" / "chat_sessions" / f"{session_id}.json"
    if not path.exists():
        return jsonify([])
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        visible = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in raw
            if msg.get("role") in ("user", "assistant") and msg.get("content")
        ]
        return jsonify(visible)
    except Exception:
        return jsonify([])


@app.route("/api/chat/<session_id>/clear", methods=["POST"])
def api_chat_clear(session_id):
    path = REPO_ROOT / "single_stock_research" / "chat_sessions" / f"{session_id}.json"
    if path.exists():
        path.unlink()
    return jsonify({"ok": True})


@app.route("/api/schedule", methods=["POST"])
def api_create_schedule():
    data = request.json or {}
    time_str  = data.get("time", "21:40").strip()
    provider  = data.get("provider", "qwen")
    task_type = data.get("task_type", "short_term")   # "short_term" | "hyperscaler"
    frequency = data.get("frequency", "daily")         # "daily" | "weekly"

    try:
        h, m = map(int, time_str.split(":"))
        assert 0 <= h < 24 and 0 <= m < 60
    except Exception:
        return jsonify({"error": "Invalid time, use HH:MM"}), 400

    if provider not in ("openai", "deepseek", "qwen", "claude"):
        return jsonify({"error": "Invalid provider"}), 400
    if task_type not in ("short_term", "hyperscaler"):
        return jsonify({"error": "Invalid task_type"}), 400
    if frequency not in ("daily", "weekly"):
        return jsonify({"error": "Invalid frequency"}), 400

    tickers = []
    if task_type == "short_term":
        tickers_raw = data.get("tickers", [])
        raw_list = tickers_raw if isinstance(tickers_raw, list) else str(tickers_raw).split(",")
        for t in raw_list:
            t = t.strip().upper()
            if t and t.replace(".", "").replace("-", "").replace("^", "").isalnum():
                tickers.append(t)
        if not tickers:
            return jsonify({"error": "No valid tickers provided"}), 400

    schedule_id = f"sched_{datetime.now().strftime('%H%M%S%f')}"
    if frequency == "daily":
        next_run_dt = _next_weekday_run_dt(time_str)
        delay = (next_run_dt - datetime.now()).total_seconds()
    else:
        delay = _seconds_until(time_str)
        next_run_dt = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if next_run_dt <= datetime.now():
            next_run_dt += timedelta(days=1)

    timer = threading.Timer(delay, _execute_schedule, args=[schedule_id])
    timer.daemon = True
    timer.start()

    schedules[schedule_id] = {
        "id":        schedule_id,
        "time":      time_str,
        "tickers":   tickers,
        "provider":  provider,
        "task_type": task_type,
        "frequency": frequency,
        "status":    "scheduled",
        "next_run":  next_run_dt.strftime("%Y-%m-%d %H:%M"),
        "last_run":  None,
        "timer":     timer,
    }
    return jsonify({"schedule_id": schedule_id, "next_run": schedules[schedule_id]["next_run"]})


@app.route("/api/schedules")
def api_list_schedules():
    return jsonify([
        {k: v for k, v in s.items() if k != "timer"}
        for s in schedules.values()
    ])


@app.route("/api/schedules/<schedule_id>", methods=["DELETE"])
def api_delete_schedule(schedule_id):
    sched = schedules.get(schedule_id)
    if not sched:
        return jsonify({"error": "Not found"}), 404
    if sched.get("timer"):
        sched["timer"].cancel()
    del schedules[schedule_id]
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"Reports directory: {REPORTS_DIR}")
    all_reports = list_reports()
    n_short = sum(1 for r in all_reports if r["short_term"])
    n_long  = len(all_reports) - n_short
    print(f"Found {n_long} long-term and {n_short} short-term reports")
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
