"""Medallion pipeline for TCS risk intelligence.

bronze (MySQL raw docs) -> silver (Elasticsearch structured) -> gold (ES kNN vectors +
graph).
"""
