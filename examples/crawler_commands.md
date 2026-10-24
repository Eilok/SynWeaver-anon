# WebArena Crawler Commands

Run all commands from the project root. Replace every angle-bracket placeholder with the value for your environment and set `NEO4J_PASSWORD`. If Neo4j does not use the default connection settings, also configure `NEO4J_URL`, `NEO4J_USERNAME`, and `NEO4J_DATABASE`.

The crawler records videos in `outputs/crawler/<run_id>/videos/` by default and stores screenshots, accessibility trees, logs, and the manifest in the same standard run directory. The `--site` profile supplies the namespace and site-specific crawler defaults; add a CLI option only when you intentionally override the centralized profile or runtime configuration.

## Shopping

```bash
python -m cli.crawl \
  --site shopping \
  --url "<SHOPPING_URL>" \
  --headless \
  --storage-state-path "<SHOPPING_STORAGE_STATE_PATH>"
```

`<SHOPPING_STORAGE_STATE_PATH>` must point to a valid local Playwright storage state file.

## Shopping Admin / CMS

The login hook reads the administrator credentials from environment variables. You can specify `enable-human-review` to automatically approve the list filtering:

```bash
export SHOPPING_ADMIN_USERNAME="<username>"
export SHOPPING_ADMIN_PASSWORD="<password>"

python -m cli.crawl \
  --site shopping_admin \
  --url "<SHOPPING_ADMIN_URL>" \
  --headless
```

## Forum / Reddit

```bash
python -m cli.crawl \
  --site forum \
  --url "<FORUM_URL>" \
  --headless \
  --storage-state-path "<FORUM_STORAGE_STATE_PATH>"
```

`<FORUM_STORAGE_STATE_PATH>` must point to a valid local Playwright storage state file.

## GitLab

```bash
python -m cli.crawl \
  --site gitlab \
  --url "<GITLAB_URL>" \
  --headless \
  --storage-state-path "<GITLAB_STORAGE_STATE_PATH>"
```

`<GITLAB_STORAGE_STATE_PATH>` must point to a valid local Playwright storage state file.

## Map

```bash
python -m cli.crawl \
  --site map \
  --url "<MAP_URL>" \
  --headless \
  --no-human-review
```

## Explicit Overrides

Use the shared canonical options when a run needs to differ from its site profile or centralized configuration:

```bash
python -m cli.crawl \
  --site shopping \
  --url "<SHOPPING_URL>" \
  --output-root "<OUTPUT_ROOT>" \
  --log-dir "<LOG_DIR>" \
  --max-depth 3 \
  --max-states 1000 \
  --no-human-review
```

Boolean overrides are symmetric: use `--headless` or `--no-headless`, `--human-review` or `--no-human-review`, and `--aggressive-url-dedup` or `--no-aggressive-url-dedup` as needed. `--output-root` is the standard output root; do not pass an old underscore-style option.

## Output Path Constraint

> **Warning:** Do not move, rename, or change the standard crawler run directory `outputs/crawler/<run_id>/` or any of its artifact subdirectories. The website map records paths to upstream files such as screenshots and accessibility trees. UI dataset synthesis and task synthesis use those recorded paths to locate the files; changing the directory location or naming will break downstream artifact lookup.
