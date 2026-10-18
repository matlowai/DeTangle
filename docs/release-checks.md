# Release checks

Run the auditor from the clean DeTangle repository before committing or publishing:

```bash
python -m pip install Pillow
python tools/audit_release.py
python -m unittest discover -s tests -p 'test_release.py' -v
```

The default checks tracked working files, staged Git blobs, and untracked files not excluded by Git. Ignored build directories and package metadata are not release candidates; force-adding them makes them fail the check. Auditing the index also catches a private path staged before an unstaged cleanup.

The auditor rejects model weights and serialized arrays, reference images, databases, logs, caches, build products, symlinks and unapproved binary files. Image approval is an explicit filename-and-content-hash catalog covering the curated display examples and beach figure. A replacement image cannot inherit approval merely by using the same filename. EXIF, PNG text, comments, unknown JPEG application blocks and other embedded metadata are rejected. The ordinary JPEG JFIF container with no thumbnail is allowed.

Release prose, data, configuration and source examples are checked for personal absolute paths, local file URLs, Windows drive paths and network-share paths. External web URLs, relative paths, generic path placeholders and path-detection code remain valid. One existing synthetic privacy-test fixture is exempted only in its named test module; real local paths elsewhere in tests still fail. Findings identify the release-relative file and line without printing the private text.

After creating a static DeltaScope export, audit that payload too:

```bash
python tools/audit_release.py --public-dir site
```

This checks every exported file, including hidden files. Static images must derive from the approved source assets; the check reproduces the exporter's metadata-free encoding with the installed Pillow version. Export and audit with the same image environment. A different encoder can require regenerating the static output; do not bypass a hash failure by approving unknown content.

Exit status is 0 for a clean payload, 1 for findings, and 2 when the audit cannot run. `--json` returns a machine-readable report. Adding new scientific examples requires reviewing those images and deliberately updating the approval catalog. Passing this check does not publish the site, approve additional source data, or replace scientific and visual review.
