"""FastAPI serving layer for the deepfake-detection web app.

Thin HTTP wrapper over :mod:`forge_detect.serving_infer`. Run with::

    make serve            # uv run uvicorn serving.app:app --reload

This package intentionally holds no inference logic — only request
parsing and response serialisation. All model work lives in the
research core so it stays testable and mypy-checked.
"""
