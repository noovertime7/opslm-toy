import argparse
import hashlib
import json
import re
import threading
import time

from collections import deque
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
)
from urllib.robotparser import RobotFileParser

import requests
import trafilatura

from bs4 import BeautifulSoup


# ============================================================
# Project Root
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# User Agent
# ============================================================

USER_AGENT = (
    "OpsLM-Toy-ResearchCrawler/0.3 "
    "(educational language-model dataset builder)"
)


# ============================================================
# Thread Local
#
# requests.Session 不在多个线程之间共享
# 每个线程自己创建 Session
# ============================================================

THREAD_LOCAL = threading.local()


def get_session():

    if not hasattr(
        THREAD_LOCAL,
        "session",
    ):

        session = requests.Session()

        session.headers.update(
            {
                "User-Agent":
                    USER_AGENT,

                "Accept":
                    "text/html,"
                    "application/xhtml+xml",
            }
        )

        THREAD_LOCAL.session = (
            session
        )

    return THREAD_LOCAL.session


# ============================================================
# Skip Extensions
# ============================================================

SKIP_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".mp4",
    ".mp3",
    ".wav",
    ".woff",
    ".woff2",
    ".ttf",
    ".css",
    ".js",
    ".xml",
}


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="data/phase2/sources.json",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/phase2/raw/"
            "documents.jsonl"
        ),
    )

    parser.add_argument(
        "--target-mb",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
    )

    # ========================================================
    # 并发线程数
    # ========================================================

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    # ========================================================
    # 同一域名两次请求开始之间的最小间隔
    #
    # 0.25 秒 ≈ 单域名最多约 4 req/s
    # ========================================================

    parser.add_argument(
        "--default-delay",
        type=float,
        default=0.25,
    )

    return parser.parse_args()


# ============================================================
# Path
# ============================================================

def resolve_path(
    value: str,
) -> Path:

    path = Path(
        value
    )

    if path.is_absolute():
        return path

    return (
        ROOT
        /
        path
    )


# ============================================================
# Normalize URL
# ============================================================

def normalize_url(
    base_url: str,
    href: str,
):

    try:

        url = urljoin(
            base_url,
            href,
        )

        parsed = urlparse(
            url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:

            return None

        path_lower = (
            parsed.path.lower()
        )

        for extension in SKIP_EXTENSIONS:

            if path_lower.endswith(
                extension
            ):

                return None

        # 去掉 query 和 fragment
        normalized = urlunparse(
            (
                parsed.scheme,
                parsed.netloc.lower(),
                parsed.path,
                "",
                "",
                "",
            )
        )

        return normalized

    except Exception:

        return None


# ============================================================
# URL Allowed
# ============================================================

def url_allowed(
    url: str,
    source: dict,
) -> bool:

    parsed = urlparse(
        url
    )

    hostname = (
        parsed.hostname
        or ""
    ).lower()

    allowed_domains = {
        domain.lower()
        for domain
        in source[
            "allowed_domains"
        ]
    }

    if hostname not in allowed_domains:

        return False

    prefixes = source.get(
        "path_prefixes",
        [],
    )

    if prefixes:

        matched = any(
            parsed.path.startswith(
                prefix
            )
            for prefix
            in prefixes
        )

        if not matched:

            return False

    return True


# ============================================================
# Text Normalize
# ============================================================

def normalize_text(
    text: str,
) -> str:

    text = (
        text
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .replace(
            "\u00a0",
            " ",
        )
    )

    lines = []

    previous_blank = False

    for line in text.splitlines():

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if not line:

            if not previous_blank:

                lines.append("")

            previous_blank = True

            continue

        previous_blank = False

        lines.append(
            line
        )

    return "\n".join(
        lines
    ).strip()


# ============================================================
# Hash
# ============================================================

def text_hash(
    text: str,
) -> str:

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Per-Domain Rate Limiter
#
# 即使多线程，同一个域名也限制请求速率
# ============================================================

class DomainRateLimiter:

    def __init__(self):

        self.last_request = {}
        self.locks = {}
        self.master_lock = (
            threading.Lock()
        )

    def _get_lock(
        self,
        domain,
    ):

        with self.master_lock:

            if domain not in self.locks:

                self.locks[
                    domain
                ] = threading.Lock()

            return self.locks[
                domain
            ]

    def wait(
        self,
        url,
        delay,
    ):

        domain = (
            urlparse(url)
            .netloc
            .lower()
        )

        lock = self._get_lock(
            domain
        )

        with lock:

            now = time.monotonic()

            last = self.last_request.get(
                domain,
                0.0,
            )

            remaining = (
                delay
                -
                (
                    now
                    -
                    last
                )
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )

            self.last_request[
                domain
            ] = time.monotonic()


RATE_LIMITER = (
    DomainRateLimiter()
)


# ============================================================
# Robots
# ============================================================

class RobotsCache:

    def __init__(
        self,
        timeout,
    ):

        self.timeout = timeout

        self.cache = {}

        self.lock = threading.Lock()

    def allowed(
        self,
        url,
    ):

        parsed = urlparse(
            url
        )

        root = (
            f"{parsed.scheme}://"
            f"{parsed.netloc}"
        )

        with self.lock:

            if root not in self.cache:

                robots_url = (
                    root
                    +
                    "/robots.txt"
                )

                parser = (
                    RobotFileParser()
                )

                parser.set_url(
                    robots_url
                )

                try:

                    response = requests.get(
                        robots_url,
                        timeout=self.timeout,
                        headers={
                            "User-Agent":
                                USER_AGENT,
                        },
                    )

                    if (
                        response.status_code
                        ==
                        200
                    ):

                        parser.parse(
                            response.text
                            .splitlines()
                        )

                        self.cache[
                            root
                        ] = parser

                    else:

                        self.cache[
                            root
                        ] = None

                except requests.RequestException:

                    self.cache[
                        root
                    ] = None

            parser = self.cache[
                root
            ]

        # robots.txt 获取失败时，
        # 不额外假设禁止
        if parser is None:

            return True

        return parser.can_fetch(
            USER_AGENT,
            url,
        )


# ============================================================
# Fetch
# ============================================================

def fetch_page(
    url,
    timeout,
    delay,
):

    session = get_session()

    last_error = None

    for attempt in range(3):

        try:

            # ================================================
            # 控制同域名请求频率
            # ================================================

            RATE_LIMITER.wait(
                url,
                delay,
            )

            response = session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
            )

            if response.status_code != 200:

                return None

            content_type = (
                response.headers
                .get(
                    "Content-Type",
                    "",
                )
                .lower()
            )

            if (
                "text/html"
                not in content_type
            ):

                return None

            return response

        except requests.RequestException as error:

            last_error = error

            time.sleep(
                1.0
                *
                (
                    attempt + 1
                )
            )

    if last_error:

        print(
            f"[error] {url}: "
            f"{last_error}",
            flush=True,
        )

    return None


# ============================================================
# Extract Text
# ============================================================

def extract_text(
    html,
):

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=False,
        include_images=False,
        deduplicate=True,
        favor_precision=True,
        output_format="txt",
    )

    if text:

        return normalize_text(
            text
        )

    # ========================================================
    # Fallback
    # ========================================================

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
        ]
    ):

        tag.decompose()

    main = (
        soup.find("main")
        or
        soup.find("article")
        or
        soup.body
    )

    if main is None:

        return ""

    return normalize_text(
        main.get_text(
            "\n"
        )
    )


# ============================================================
# Title
# ============================================================

def extract_title(
    html,
):

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    h1 = soup.find(
        "h1"
    )

    if h1:

        return h1.get_text(
            " ",
            strip=True,
        )

    if soup.title:

        return soup.title.get_text(
            " ",
            strip=True,
        )

    return ""


# ============================================================
# Links
# ============================================================

def extract_links(
    html,
    current_url,
    source,
):

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    result = []

    local_seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):

        url = normalize_url(
            current_url,
            anchor[
                "href"
            ],
        )

        if not url:

            continue

        if url in local_seen:

            continue

        if not url_allowed(
            url,
            source,
        ):

            continue

        local_seen.add(
            url
        )

        result.append(
            url
        )

    return result


# ============================================================
# Existing Dataset
# ============================================================

def load_existing(
    output_path,
):

    seen_urls = set()

    seen_hashes = set()

    total_bytes = 0

    documents = 0

    if not output_path.exists():

        return (
            seen_urls,
            seen_hashes,
            total_bytes,
            documents,
        )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            try:

                item = json.loads(
                    line
                )

            except Exception:

                continue

            url = item.get(
                "url"
            )

            if url:

                seen_urls.add(
                    url
                )

            digest = item.get(
                "sha256"
            )

            if digest:

                seen_hashes.add(
                    digest
                )

            text = item.get(
                "text",
                "",
            )

            total_bytes += len(
                text.encode(
                    "utf-8"
                )
            )

            documents += 1

    return (
        seen_urls,
        seen_hashes,
        total_bytes,
        documents,
    )


# ============================================================
# Worker
# ============================================================

def crawl_one_url(
    url,
    source,
    robots,
    timeout,
    min_chars,
    delay,
):

    # ========================================================
    # Robots
    # ========================================================

    if not robots.allowed(
        url
    ):

        return {
            "status":
                "robots",

            "url":
                url,

            "links":
                [],
        }

    # ========================================================
    # HTTP
    # ========================================================

    response = fetch_page(
        url=url,
        timeout=timeout,
        delay=delay,
    )

    if response is None:

        return {
            "status":
                "failed",

            "url":
                url,

            "links":
                [],
        }

    final_url = normalize_url(
        response.url,
        response.url,
    )

    if not final_url:

        return {
            "status":
                "failed",

            "url":
                url,

            "links":
                [],
        }

    if not url_allowed(
        final_url,
        source,
    ):

        return {
            "status":
                "outside",

            "url":
                url,

            "links":
                [],
        }

    html = response.text

    # ========================================================
    # 先发现链接
    # ========================================================

    links = extract_links(
        html=html,
        current_url=final_url,
        source=source,
    )

    # ========================================================
    # 正文
    # ========================================================

    text = extract_text(
        html
    )

    if len(text) < min_chars:

        return {
            "status":
                "short",

            "url":
                final_url,

            "links":
                links,
        }

    digest = text_hash(
        text
    )

    title = extract_title(
        html
    )

    return {

        "status":
            "ok",

        "url":
            final_url,

        "title":
            title,

        "text":
            text,

        "sha256":
            digest,

        "links":
            links,

    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.workers < 1:

        raise ValueError(
            "workers must be >= 1"
        )

    config_path = resolve_path(
        args.config
    )

    output_path = resolve_path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    sources = config[
        "sources"
    ]

    # ========================================================
    # Existing
    # ========================================================

    (
        seen_urls,
        seen_hashes,
        total_bytes,
        total_documents,
    ) = load_existing(
        output_path
    )

    target_bytes = int(
        args.target_mb
        *
        1024
        *
        1024
    )

    robots = RobotsCache(
        timeout=args.timeout
    )

    print(
        "=" * 80
    )

    print(
        "OpsLM Phase-2 Concurrent Crawler"
    )

    print(
        "=" * 80
    )

    print(
        "Workers:",
        args.workers
    )

    print(
        "Target:",
        f"{args.target_mb:.1f} MB"
    )

    print(
        "Existing documents:",
        total_documents
    )

    print(
        "Existing text:",
        f"{total_bytes / 1024 / 1024:.2f} MB"
    )

    print()

    # ========================================================
    # Output
    # ========================================================

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as output:

        # ====================================================
        # Source By Source
        #
        # 每个 Source 内部多线程
        # ====================================================

        for source in sources:

            if total_bytes >= target_bytes:

                break

            source_name = source[
                "name"
            ]

            max_pages = int(
                source.get(
                    "max_pages",
                    1000,
                )
            )

            # =================================================
            # Source 可以单独控制 delay
            #
            # sources.json 里如果还是 0.5，
            # 就按 0.5
            # =================================================

            delay = float(
                source.get(
                    "delay_seconds",
                    args.default_delay,
                )
            )

            # =================================================
            # Source 也可以覆盖 worker 数
            # =================================================

            source_workers = int(
                source.get(
                    "workers",
                    args.workers,
                )
            )

            print()

            print(
                "=" * 80
            )

            print(
                "Source:",
                source_name
            )

            print(
                "License:",
                source.get(
                    "license",
                    "unknown",
                )
            )

            print(
                "Workers:",
                source_workers
            )

            print(
                "Domain request interval:",
                f"{delay:.2f}s"
            )

            print(
                "=" * 80
            )

            # =================================================
            # Frontier
            # =================================================

            frontier = deque()

            queued = set()

            processed = set(
                seen_urls
            )

            for seed in source[
                "seed_urls"
            ]:

                seed = normalize_url(
                    seed,
                    seed,
                )

                if (
                    seed
                    and
                    seed not in processed
                ):

                    frontier.append(
                        seed
                    )

                    queued.add(
                        seed
                    )

            fetched_pages = 0

            saved_pages = 0

            # =================================================
            # Thread Pool
            # =================================================

            with ThreadPoolExecutor(
                max_workers=source_workers
            ) as executor:

                while (
                    frontier
                    and
                    fetched_pages < max_pages
                    and
                    total_bytes < target_bytes
                ):

                    # =========================================
                    # 一次保持 2 倍 worker 的任务在途
                    # =========================================

                    batch_urls = []

                    batch_limit = (
                        source_workers
                        *
                        2
                    )

                    while (
                        frontier
                        and
                        len(batch_urls)
                        <
                        batch_limit
                        and
                        (
                            fetched_pages
                            +
                            len(batch_urls)
                        )
                        <
                        max_pages
                    ):

                        url = frontier.popleft()

                        if url in processed:

                            continue

                        processed.add(
                            url
                        )

                        batch_urls.append(
                            url
                        )

                    if not batch_urls:

                        continue

                    futures = {

                        executor.submit(
                            crawl_one_url,
                            url,
                            source,
                            robots,
                            args.timeout,
                            args.min_chars,
                            delay,
                        ):
                        url

                        for url
                        in batch_urls

                    }

                    for future in as_completed(
                        futures
                    ):

                        fetched_pages += 1

                        original_url = futures[
                            future
                        ]

                        try:

                            result = future.result()

                        except Exception as error:

                            print(
                                "[worker-error]",
                                original_url,
                                error,
                                flush=True,
                            )

                            continue

                        # =====================================
                        # Links 加入 Frontier
                        # =====================================

                        for link in result.get(
                            "links",
                            [],
                        ):

                            if link in processed:

                                continue

                            if link in queued:

                                continue

                            frontier.append(
                                link
                            )

                            queued.add(
                                link
                            )

                        # =====================================
                        # 正文
                        # =====================================

                        if (
                            result.get(
                                "status"
                            )
                            !=
                            "ok"
                        ):

                            continue

                        final_url = result[
                            "url"
                        ]

                        digest = result[
                            "sha256"
                        ]

                        # URL 去重
                        if final_url in seen_urls:

                            continue

                        # 正文去重
                        if digest in seen_hashes:

                            seen_urls.add(
                                final_url
                            )

                            continue

                        text = result[
                            "text"
                        ]

                        document = {

                            "source":
                                source_name,

                            "url":
                                final_url,

                            "title":
                                result.get(
                                    "title",
                                    "",
                                ),

                            "license":
                                source.get(
                                    "license",
                                    "unknown",
                                ),

                            "sha256":
                                digest,

                            "text":
                                text,

                        }

                        # =====================================
                        # JSONL
                        # =====================================

                        output.write(
                            json.dumps(
                                document,
                                ensure_ascii=False,
                            )
                            +
                            "\n"
                        )

                        output.flush()

                        # =====================================
                        # State
                        # =====================================

                        seen_urls.add(
                            final_url
                        )

                        seen_hashes.add(
                            digest
                        )

                        saved_pages += 1

                        total_documents += 1

                        text_bytes = len(
                            text.encode(
                                "utf-8"
                            )
                        )

                        total_bytes += (
                            text_bytes
                        )

                        print(

                            f"[saved] "
                            f"source="
                            f"{source_name:<20} "
                            f"docs="
                            f"{total_documents:<6} "
                            f"size="
                            f"{total_bytes / 1024 / 1024:8.2f} MB "
                            f"frontier="
                            f"{len(frontier):<6} "
                            f"url={final_url}",

                            flush=True,

                        )

                        if total_bytes >= target_bytes:

                            break

            print()

            print(
                f"{source_name}: "
                f"fetched="
                f"{fetched_pages}, "
                f"saved="
                f"{saved_pages}"
            )

    # ========================================================
    # Finished
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "Crawl Finished"
    )

    print(
        "=" * 80
    )

    print(
        "Documents:",
        total_documents
    )

    print(
        "Text size:",
        f"{total_bytes / 1024 / 1024:.2f} MB"
    )

    print(
        "Output:",
        output_path
    )


if __name__ == "__main__":

    main()
