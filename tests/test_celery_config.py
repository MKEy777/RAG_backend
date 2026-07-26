from __future__ import annotations

from knowlink.tasks.celery_app import celery_app


def test_queue_routing():
    routes = celery_app.conf.task_routes
    assert routes["knowlink.tasks.parse_tasks.parse_and_index_task"]["queue"] == "ingest"
    assert celery_app.conf.task_default_queue == "enrich"


def test_time_limits():
    assert celery_app.conf.task_soft_time_limit == 1500
    assert celery_app.conf.task_time_limit == 1800
