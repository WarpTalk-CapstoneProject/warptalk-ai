import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "metrics_exporter.app:app",
        host="0.0.0.0",
        port=int(os.getenv("METRICS_PORT", "9108")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
