# Troubleshooting

| Symptom | What to check |
| --- | --- |
| `uv` unavailable | Install uv and reopen your terminal. If you installed it with pip, `python -m uv` also works. |
| Python version mismatch | Run `uv python install 3.11`, then `uv sync`. |
| Raw CSV missing or empty | Download the IBM Telco sample using the README link and use the expected filename. |
| Invalid data or duplicate customer ID | Check the named field and resolve conflicting records in the source before training. |
| API returns 422 | Check field names, valid categories, finite non-negative charges and integer tenure. See the API's `/docs` page. |
| API returns 503 | Run `uv run python train.py` and check that the model files and reports were written. |
| UI cannot connect | Start Uvicorn and check the API port and `CHURNGUARD_API_URL`. |
| TensorFlow fails to import on Windows | Verify that uv selected 64-bit Python 3.11 and that the Microsoft Visual C++ runtime is installed. |
| Notebook import fails | Launch Jupyter from the repository root using `uv run jupyter notebook` and select its Python kernel. |

Stop any active demo before replacing its artifacts with a fresh training run. Use standard mode to generate the full tuning comparison after rehearsing with quick mode.
