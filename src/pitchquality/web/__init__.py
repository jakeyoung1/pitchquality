"""Live API and dashboard on top of ``data/pitchquality.db``.

This subpackage deliberately imports nothing from the modeling code, so the
deployed service needs only FastAPI and Jinja, not pandas or scikit-learn.
"""
