# Test evidence

| File | Contents |
|---|---|
| `pytest_output.txt` | Full `pytest -v tests/` output (13 October 2026) |
| `browser_checks.txt` | Log of the browser click-through checks run on 11 October 2026: the record cited in the dissertation |
| `timings.csv` | Stage timings on the 5,000-row Home Credit sample, both attributes (11 October 2026) |
| `browser_checks.py` | The script that ran the browser checks |
| `make_german_upload.py` | Rebuilds `german_with_lender_score.csv`, the upload file the browser checks need |

`browser_checks.py` is the 11 October script with its paths made portable. Its checks are unchanged; it also saves screenshots 11 and 12, which were taken separately on 13 October.

## How to re-run the browser checks

1. **Install.** The project environment (as in the main README), plus the WebSocket client the script uses to talk to Chrome, and Google Chrome or Chromium:

   ```bash
   conda create -n creditfair python=3.11
   conda activate creditfair
   pip install -r requirements.txt
   pip install websocket-client
   ```

2. **Data.** Download the three datasets into `data/` as described in [`data/README.md`](../../data/README.md). The sample audit's applicant lookup also needs the stored example applicants, built once (about 12 minutes):

   ```bash
   python app/build_demo_borrowers.py
   ```

3. **Upload file.** Write `data/german_with_lender_score.csv` (git ignores it):

   ```bash
   python docs/test_evidence/make_german_upload.py
   ```

4. **Start the dashboard on port 8502**, in its own terminal, and leave it running:

   ```bash
   streamlit run app/dashboard.py --server.port 8502 --server.headless true
   ```

5. **Run the checks** from the repository root, in a second terminal (about 7 minutes on an Apple M1):

   ```bash
   python docs/test_evidence/browser_checks.py
   ```

   `--parts sample upload` runs only some parts; `--help` lists every option. If Chrome is not found, pass `--chrome /path/to/chrome`.

6. **Results** go to a new folder, `docs/test_evidence/run_<date>_<time>/`:
   - `browser_checks.log`: one PASS or FAIL line per check, in the same format as `browser_checks.txt`, with a final "N of M checks passed" line;
   - `browser_checks.jsonl`: the same results as JSON;
   - `screenshots/` and `downloads/`: what the run captured and downloaded.

   The script exits with code 0 only if every check passes. Neither the committed screenshots in `docs/screenshots/` nor `browser_checks.txt` are touched; a run folder is not committed unless you add it.

The dashboard caches results while it runs, so a second run against the same dashboard is faster. For timings comparable with the 11 October record, restart the dashboard first.
