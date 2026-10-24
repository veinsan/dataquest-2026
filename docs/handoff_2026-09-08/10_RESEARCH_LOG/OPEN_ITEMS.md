# Open Items

1. Choose and record an exact audited TALENT Git commit for V12.
2. Ensure all TALENT declared dependencies are installed and version-compatible, especially `qhoptim>=1.1.0` and torch>=2.8.
3. Freeze/verify the eight V12 ModernNCA configurations from the official HPO space and HPO seed 20260908.
4. Static-audit the entire V12 worker/downstream path after source pinning: preprocessing, HPO, two-seed refit, model persistence/resume-test, Stage-S artifact routing, full-CV/test aggregation, bootstrap, thresholds, blend, submission gate.
5. Run Stage R only after static audit.
6. Run Stage S folds 0/2 only after Stage R.
7. Do not run full CV if no Stage-S gate passes.
8. If V12 is scientifically negative, re-open hypothesis selection rather than doing V12 micro-rescues. Likely next research space: latent/psychometric representation; TabR only if learned-local geometry evidence justifies it.
9. Preserve current recovery submission as champion/reference.
