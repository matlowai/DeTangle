# DeltaScope site and GitHub Pages

The Pages build includes the beach and four approved creative scenes, the research guide,
all thirteen method explanations, broader aggregate findings in Dig more, lessons and local scoring
instructions. It is a static version of DeltaScope. Scoring runs in the Python
app on the operator's machine.

## Build and preview locally

From the repository root:

```bash
python -m pip install './lab[images]'
python tools/audit_release.py
deltascope export lab/src/deltascope/case_studies/conditional_identity.json --out _site
python -m http.server 8891 --bind 127.0.0.1 --directory _site
```

Open `http://127.0.0.1:8891`. Export refuses an existing destination; choose a
fresh directory for the next build. Relative asset URLs support a project path
such as `DeTangle/`, and the export also works directly from its `index.html`.

The image scores refer to native renders, whose hashes remain in the manifest.
The distributed images are resized display copies. Export strips image
metadata, excludes reference images and evaluator internals, validates image
hashes, and refuses stale measurements. It never includes a checkpoint or
enables a scoring endpoint.

## Hosted site and deployment

The site is hosted at [matlowai.github.io/DeTangle](https://matlowai.github.io/DeTangle/).
It is built from this repository through GitHub Actions. Scoring remains local.

Publishing is manual: **Actions → DeltaScope Pages preview → Run workflow**.
Leave `deploy` unchecked to build a review artifact, or check it to publish.
The repository variable `ENABLE_PAGES` must also be `true`. Pushing code alone
does not deploy the site. Review changed images, prompts and dataset contents
before a publishing run.

For another installation, configure the repository's Pages source as
**GitHub Actions**, set `ENABLE_PAGES` to `true`, and run the workflow with
`deploy` checked. No domain or Pages setting is changed by a local build.

A private repository does not by itself provide a private Pages website. Check
the intended audience and the hosting plan when activating it. See GitHub's
[custom workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [Pages access controls](https://docs.github.com/en/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site).
