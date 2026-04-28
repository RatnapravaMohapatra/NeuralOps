import os
import uvicorn


def run():
    env = os.getenv("ENV", "dev").lower()

    is_dev = env == "dev"

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=is_dev,  # ✅ only in dev
        workers=1 if is_dev else int(os.getenv("WORKERS", 2)),
        log_level="info",
    )


if __name__ == "__main__":
    run()
