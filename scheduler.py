from apscheduler.schedulers.background import BackgroundScheduler
import requests

def keep_server_awake():
    try:
        # 🔴 CHANGE THIS to your deployed URL
        url = "https://depression-classifier.onrender.com/health"
        response = requests.get(url, timeout=10)
        print(f"[CRON] Server pinged: {response.status_code}")
    except Exception as e:
        print("[CRON] Ping failed:", e)

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        keep_server_awake,
        "interval",
        minutes=1
    )
    scheduler.start()
    return scheduler
