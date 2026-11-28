Run ./release.sh in these four modes and record, one per line, the mode,
the exit code and the first line of stderr (or stdout on success):
  --reuse-capabilities
  --missing-artifact
  --hash-mismatch
  --apply   (create an empty artifact.bin first so this one succeeds)
Write the four lines to result.txt as 'MODE|EXIT|LINE'. Do not modify
release.sh.
