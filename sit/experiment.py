"""Experiment runner for the SIT experiment.

Executes the sampling loop: for each (prompt, schema variant, model)
combination, calls the API 20 times at temperature 1.0 and stores
the structured responses to disk. Resumable if interrupted.

50 prompts x 3 variants x 2 models x 20 samples = 6,000 calls.
"""
