from __future__ import annotations

from knowlink.tasks.celery_app import celery_app


def main():
    celery_app.worker_main(["worker", "--pool=solo", "--loglevel=info"])


if __name__ == "__main__":
    main()
