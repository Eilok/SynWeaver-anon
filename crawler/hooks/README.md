# 站点扩展 Hook

crawler 核心不包含特定网站逻辑。需要登录或覆盖表单值时，通过 `--login-script PATH` 加载用户脚本；未提供脚本时保持普通未登录爬取行为。

## 登录 Hook

脚本可定义 `login(browser_session)`：

```python
def login(browser_session):
    page = browser_session.page
    page.get_by_placeholder("user name").fill("admin")
    page.get_by_placeholder("password").fill("password")
    page.get_by_role("button", name="Sign in").click()
    browser_session._wait_for_stability()
```

运行方式：

```bash
python -m cli.crawl \
  --url "http://localhost:7780/admin" \
  --neo4j-namespace "shopping_admin" \
  --login-script ./my_site_login.py
```

`login()` 会在首次打开起始 URL 后调用，并在回溯重放重建浏览器上下文后再次调用，因此实现必须自包含且可重复执行。真实凭据应从环境变量读取，不要写入仓库。

也可先运行 `python -m cli.login` 生成 Playwright storage state，再通过 `--storage-state-path` 使用。

## 表单值覆盖

同一脚本可选定义 `override_form_value(field, default_value)`：

```python
def override_form_value(field, default_value):
    if field.get("xpath") == "/html/body/.../input[1]":
        return "restaurant"
    return None
```

返回字符串表示覆盖 LLM 生成值；返回 `None` 表示沿用默认值。可复制 `login_script_template.py` 作为起点。
