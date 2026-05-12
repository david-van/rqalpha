"""
小鹅通专栏文章批量下载脚本。

用法：直接修改文件末尾 download() 的入参，然后运行 python download_xiaoe.py。

使用前请修改 COOKIE 为你自己的登录 Cookie。
"""
import os
import random
import re
import time
import requests

# ============ 配置区 ============

COLUMN_ID = "p_5cb85a80f0082_VN38N4AW"
PRODUCT_ID = "p_5cb85a80f0082_VN38N4AW"
BASE_DIR = "xiaoe_articles"

# 从浏览器复制你的 Cookie 字符串填在这里
COOKIE = (
    "xenbyfpfUnhLsdkZbX=0; newuserdays=90; olduserdays=180; "
    "regtime=1610063193; shop_version_type=4; colla_login=1; "
    "sensorsdata2015jssdkcross=%7B%22%24device_id%22%3A%2219e174e7bfa9f4-030b874bafcb2b-"
    "26061151-2073600-19e174e7bfbc93%22%7D; "
    "sajssdk_2015_new_user_app0tgi74k25140_h5_xiaoeknow_com=1; "
    "sa_jssdk_2015_app0tgi74k25140_h5_xiaoeknow_com=%7B%22distinct_id%22%3A%22u_5ff79d594c3d8"
    "_3Q1k3xsdLh%22%2C%22first_id%22%3A%2219e174e7bfa9f4-030b874bafcb2b-26061151-2073600-"
    "19e174e7bfbc93%22%2C%22props%22%3A%7B%7D%7D; "
    "xiaoe_loading_show=1; "
    "ko_token=2a246a663c93ed2298c9bb90c1ae95a4; "
    "logintime=1778623701"
)

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/x-www-form-urlencoded",
    "referer": f"https://app0tgi74k25140.h5.xiaoeknow.com/p/course/column/{COLUMN_ID}",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
}

BASE_URL = "https://app0tgi74k25140.h5.xiaoeknow.com"
LIST_API = f"{BASE_URL}/xe.course.business.column.items.get/2.0.0"
DETAIL_API = f"{BASE_URL}/xe.course.business_go.get.detail/2.0.0"

PAGE_SIZE = 20
SLEEP_MIN = 1.0
SLEEP_MAX = 10.0
DOWNLOAD_IMAGES = True
IMAGES_DIR = "images"

# ============ 脚本逻辑 ============

session = requests.Session()
session.headers.update(HEADERS)
for item in COOKIE.split("; "):
    if "=" in item:
        key, value = item.split("=", 1)
        session.cookies.set(key, value)


def fetch_article_list(page_index: int) -> dict:
    data = {
        "bizData[column_id]": COLUMN_ID,
        "bizData[page_index]": page_index,
        "bizData[page_size]": PAGE_SIZE,
        "bizData[sort]": "asc",
    }
    resp = session.post(LIST_API, data=data, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result["code"] != 0:
        raise RuntimeError(f"列表 API 报错: {result}")
    return result["data"]


def fetch_article_detail(resource_id: str) -> dict:
    data = {
        "bizData[resource_id]": resource_id,
        "bizData[product_id]": PRODUCT_ID,
    }
    resp = session.post(DETAIL_API, data=data, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result["code"] != 0:
        raise RuntimeError(f"详情 API 报错 (resource_id={resource_id}): {result}")
    return result["data"]


def sanitize_filename(title: str, date: str = "") -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)
    safe = safe.strip().replace("\n", " ").replace("\r", "")
    if len(safe) > 80:
        safe = safe[:80]
    if date:
        date = date.replace(".", "-")
        return f"{date}_{safe}.html"
    return f"{safe}.html"


def build_html(title: str, content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ max-width: 800px; margin: 0 auto; padding: 20px; font-family: -apple-system, "Microsoft YaHei", sans-serif; line-height: 1.8; }}
  img {{ max-width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ccc; padding: 8px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{content}
</body>
</html>"""


IMG_RE = re.compile(r'<img\s[^>]*?src="([^"]+)"[^>]*?>')
ATTR_ALT_RE = re.compile(r'alt="([^"]*)"')
ATTR_TITLE_RE = re.compile(r'title="([^"]*)"')
TITLE_DATE_RE = re.compile(
    r'(20\d{2})'           # 年份
    r'[-./]?'              # 可选分隔符
    r'(\d{1,2})'           # 月份
    r'[-./]?'              # 可选分隔符
    r'(\d{1,2})'           # 日期
)


def extract_date_from_title(title: str) -> str | None:
    """从标题中提取日期，返回 YYYY-MM-DD 格式；未找到返回 None。"""
    m = TITLE_DATE_RE.search(title)
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), m.group(3)
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def resolve_date(title: str, start_at: str) -> str:
    """优先从标题提取日期，提取不到则用 start_at。返回 YYYY-MM-DD 格式。"""
    extracted = extract_date_from_title(title)
    if extracted:
        return extracted
    return start_at.replace(".", "-") if start_at else ""


def get_year_dir(date_str: str) -> str:
    """从日期字符串提取年份，返回 {BASE_DIR}/{year} 目录。"""
    year = date_str[:4] if date_str and len(date_str) >= 4 else "unknown"
    return os.path.join(BASE_DIR, year)


def ensure_output_dir(date_str: str) -> str:
    """创建并返回日期对应的年份输出目录。"""
    d = get_year_dir(date_str)
    os.makedirs(d, exist_ok=True)
    return d


def download_images(content: str, date: str, output_dir: str) -> str:
    """下载文章中的图片到本地，替换 src 为本地路径。返回替换后的 content。"""
    if not DOWNLOAD_IMAGES:
        return content

    image_dir = os.path.join(output_dir, IMAGES_DIR)
    os.makedirs(image_dir, exist_ok=True)

    date = date.replace(".", "-")
    ext_only = re.compile(r'^\.\w+$')

    def _pick_name(tag: str, url: str) -> str:
        """从 img 标签中提取有意义的文件名。alt > title > URL 文件名。"""
        for attr_re in (ATTR_ALT_RE, ATTR_TITLE_RE):
            m = attr_re.search(tag)
            if m and m.group(1).strip() and not ext_only.match(m.group(1).strip()):
                return m.group(1).strip()
        name = url.rsplit("/", 1)[-1].split("?")[0]
        return name if name else "image"

    used: dict[str, int] = {}

    # 预填已有文件，实现跨文章去重
    for fname in os.listdir(image_dir):
        used[fname] = used.get(fname, 0) + 1
        m = re.match(r'(.+)_(\d+)(\.\w+)$', fname)
        if m:
            base = m.group(1) + m.group(3)
            n = int(m.group(2))
            used[base] = max(used.get(base, 0), n)

    def _replace(m: re.Match) -> str:
        tag = m.group(0)
        url = m.group(1)
        raw_name = _pick_name(tag, url)
        name = f"{date}_{raw_name}" if date else raw_name

        # 去重：同名文件加 _2, _3...
        if name in used:
            used[name] += 1
            base, ext = os.path.splitext(name)
            name = f"{base}_{used[name]}{ext}"
        else:
            used[name] = 1

        local_path = os.path.join(image_dir, name)

        if not os.path.exists(local_path):
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                print(f"  [图片下载失败] {name}: {e}")

        return m.group(0).replace(url, f"{IMAGES_DIR}/{name}")

    return IMG_RE.sub(_replace, content)


def download(
    start: int = 1,
    end: int | None = None,
    *,
    skip_existing: bool = True,
):
    """
    下载专栏文章。

    参数：
      start: 从第几篇开始（1-based）
      end:   到第几篇结束，None 表示直到最后一篇
      skip_existing: 是否跳过已存在的文件

    常用示例：
      download(1, 1)        → 只下载第 1 篇做测试
      download(5, 5)        → 只下载第 5 篇
      download(1, 10)       → 下载第 1~10 篇
      download(1, None)     → 下载全部
      download(45, 45, skip_existing=False)  → 强制重新下载第 45 篇
    """
    os.makedirs(BASE_DIR, exist_ok=True)

    print("正在获取文章总数...")
    first_page = fetch_article_list(1)
    total = first_page["total"]
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"共 {total} 篇文章，{total_pages} 页（每页 {PAGE_SIZE} 篇）")
    if end is None:
        end = total
    print(f"本次下载范围: 第 {start}~{end} 篇")
    print(f"输出目录: {BASE_DIR}/<年份>/")
    print("=" * 60)

    idx = 0
    seen: set[str] = set()
    downloaded = 0

    for page in range(1, total_pages + 1):
        if page == 1:
            page_data = first_page
        else:
            if idx >= end:
                break
            print(f"正在获取第 {page}/{total_pages} 页文章列表...")
            page_data = fetch_article_list(page)
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        for item in page_data["list"]:
            if item["resource_id"] in seen:
                continue
            seen.add(item["resource_id"])
            idx += 1

            if idx < start:
                continue
            if idx > end:
                break

            rid = item["resource_id"]
            title = item["resource_title"]

            date_str = resolve_date(title, item.get("start_at", ""))
            output_dir = ensure_output_dir(date_str)
            filename = sanitize_filename(title, date_str)
            filepath = os.path.join(output_dir, filename)
            if skip_existing and os.path.exists(filepath):
                print(f"[{idx}/{total}] 跳过（已存在）: {title}")
                downloaded += 1
                continue

            print(f"[{idx}/{total}] 下载: {title}")

            try:
                detail = fetch_article_detail(rid)
                content = detail.get("org_content", "")
                content = download_images(content, date_str, output_dir)
                html = build_html(title, content)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html)
                downloaded += 1
            except Exception as e:
                print(f"  [错误] {e}")
                continue

            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        if idx >= end:
            break

    print("=" * 60)
    print(f"完成！共下载 {downloaded} 篇文章，保存在 ./{BASE_DIR}/<年份>/ 目录")


def download_by_resource_id(resource_id: str, title: str = "", date: str = ""):
    """
    根据 resource_id 下载单篇文章。

    参数：
      resource_id: 文章资源 ID，如 'i_5cbe84eda268b_8KosNjlE'
      title: 文章标题（可选），不传则以 resource_id 作为文件名
      date: 发表日期（可选），格式如 '2019.04.23' 或 '2019-04-23'

    用法：
      download_by_resource_id("i_5cbe84eda268b_8KosNjlE")
      download_by_resource_id("i_5cbe84eda268b_8KosNjlE", "投资日志（1）", "2019.04.23")
    """
    date = resolve_date(title, date)
    output_dir = ensure_output_dir(date)
    print(f"正在下载: {resource_id}")
    detail = fetch_article_detail(resource_id)
    content = detail.get("org_content", "")
    content = download_images(content, date, output_dir)

    filename_base = title if title else resource_id
    filename = sanitize_filename(filename_base, date)
    filepath = os.path.join(output_dir, filename)

    html = build_html(title or resource_id, content)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"下载成功！文件保存到: {filepath}")


# ============ 运行入口：修改这里 ============

if __name__ == "__main__":
    download(
        start=130,
        end=170,              # None 表示全部，改成数字表示只下载到第几篇
        skip_existing=True,  # True: 跳过已存在的文件; False: 强制重新下载
    )
    # 如果知道 resource_id，也可以直接调这个：
    # download_by_resource_id("i_5d074badd1ff5_Oq3fkCWw", date='2019-06-17')
