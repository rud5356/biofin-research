#!/usr/bin/env python3
"""BIOFIN Docker command wizard (Linux, Python 3.8+; no dependencies)."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


# 환경 설정은 config.json, 실행 변수 기본값은 defaults.json에서 관리합니다.
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config(path=CONFIG_PATH):
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"설정 파일을 읽을 수 없습니다: {path} ({exc})") from exc
    if not isinstance(config, dict):
        raise ValueError(f"설정 파일은 JSON 객체여야 합니다: {path}")
    return {key: value for key, value in config.items() if not key.startswith("_comment")}


try:
    CONFIG = load_config()
    DEFAULTS = load_config(CONFIG_PATH.with_name("defaults.json"))
except ValueError as exc:
    raise SystemExit(str(exc)) from exc


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return str(default)
        print("값을 입력해주세요.")


def choose(prompt, options):
    print(f"\n{prompt}")
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}")
    while True:
        value = ask("번호", "1")
        if value.isdigit() and 1 <= int(value) <= len(options):
            return int(value)
        print("목록에 있는 번호를 입력해주세요.")


def integer(prompt, default, minimum=1):
    while True:
        value = ask(prompt, default)
        if value.isdigit() and int(value) >= minimum:
            return value
        print(f"{minimum} 이상의 정수를 입력해주세요.")


def project_path(prompt, default, root, kind):
    root = root.resolve()
    suggested = root / default
    current = suggested if kind == "dir" or suggested.is_dir() else suggested.parent
    while not current.is_dir():
        current = current.parent
    current = current.resolve()
    try:
        current.relative_to(root)
    except ValueError:
        current = root
    while True:
        try:
            entries = []
            for entry in current.iterdir():
                if entry.name.startswith("."):
                    continue
                resolved = entry.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                if entry.is_dir() or (kind == "file" and entry.is_file() and entry.suffix.lower() == ".csv"):
                    entries.append(entry)
            entries.sort(key=lambda p: (not p.is_dir(), p.name.casefold()))
        except OSError as error:
            print(f"폴더를 읽을 수 없습니다: {error}")
            if current == root:
                raise
            current = current.parent
            continue
        print(f"\n{prompt} — 현재 폴더: {current.relative_to(root).as_posix()}")
        print("  0. 상위 폴더" if current != root else "  (프로젝트 최상위 폴더)")
        if kind != "file":
            print("  s. 현재 폴더 선택")
        if kind == "output":
            print("  n. 현재 위치에 새 결과 폴더 지정")
        print("  q. 취소")
        for index, entry in enumerate(entries, 1):
            print(f"  {index}. {'[폴더] ' if entry.is_dir() else '[CSV] '}{entry.name}")
        if not entries:
            print("  선택 가능한 항목이 없습니다.")
        value = ask("번호 또는 메뉴 문자").lower()
        if value == "q":
            raise KeyboardInterrupt
        if value == "0" and current != root:
            current = current.parent
            continue
        if value == "s" and kind != "file":
            path = current
        elif value == "n" and kind == "output":
            name = ask("새 폴더 이름 (Enter: 자동 날짜 이름)", suggested.name)
            if name in (".", "..") or "/" in name or "\\" in name or "\x00" in name or Path(name).is_absolute():
                print("경로가 아닌 폴더 이름 하나만 입력해주세요.")
                continue
            path = current / name
        elif value.isdigit() and 1 <= int(value) <= len(entries):
            path = entries[int(value) - 1].resolve()
            if path.is_dir():
                current = path
                continue
        else:
            print("목록에 있는 번호 또는 메뉴 문자를 입력해주세요.")
            continue
        path = path.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            print("프로젝트 폴더 안에서 선택해주세요.")
            continue
        if kind == "output" and path.exists():
            if not path.is_dir():
                print("같은 이름의 파일이 있습니다.")
                continue
            print("기존 결과 폴더입니다. 실행 스크립트에 따라 파일이 덮어써질 수 있습니다.")
            if ask("이 폴더를 사용할까요? (y/N)", "n").lower() != "y":
                continue
        print(f"→ {prompt}: {relative.as_posix()}")
        return relative.as_posix()


REGIONS = ["서울", "부산", "대구", "대전", "인천", "광주", "울산", "세종", "경기",
           "충북", "충남", "전남", "경북", "경남", "제주", "강원", "전북"]


def dataset_name(year, region=None):
    """수집과 문서 선택에서 같은 폴더 이름 규칙을 사용한다."""
    if region is None:
        return str(year)
    suffix = CONFIG["REGION_CODES"][region]
    if not suffix or not all(c.isascii() and (c.isalnum() or c in "-_") for c in suffix):
        raise RuntimeError(f"config.json: 잘못된 지역 영문 코드: {region}")
    return f"{year}_{suffix}"


def dataset_path(year, region=None):
    base = Path(CONFIG["DOCUMENT_DIR"])
    if region is None:
        return base / "열린재정" / str(year)
    dataset_name(year, region)  # 지역 코드 검증
    return base / "지방재정" / str(year) / CONFIG["REGION_CODES"][region]


def select_document_dir(root):
    source = choose("문서 자료 구분", ["열린재정", "지방재정"])
    year = integer("문서 회계연도", DEFAULTS["CRAWL_YEAR"] or datetime.date.today().year - 1, 2000)
    region = REGIONS[choose("지역 선택", REGIONS) - 1] if source == 2 else None
    folder = root / dataset_path(year, region)
    docs = folder / "사업설명자료"
    try:
        relative = docs.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise RuntimeError("문서 폴더는 프로젝트 안에 있어야 합니다.")
    if not docs.is_dir():
        raise RuntimeError(f"사업설명자료 폴더가 없습니다: {docs}\nDOCUMENT_DIR와 REGION_CODES 설정을 확인해주세요.")
    if not any(path.is_file() and path.suffix.lower() in (".pdf", ".hwp", ".hwpx", ".txt", ".doc", ".docx")
               for path in docs.rglob("*")):
        raise RuntimeError(f"사업설명자료가 없습니다: {docs}\n목록만 수집한 경우 상세 자료를 먼저 수집해주세요.")
    print(f"→ 사업설명자료 폴더: {relative}")
    return relative


def select_csv(root, folder):
    """지정 폴더 바로 아래 CSV가 하나면 자동 선택한다."""
    directory = (root / folder).resolve()
    try:
        directory.relative_to(root.resolve())
    except ValueError:
        raise RuntimeError("CSV 폴더는 프로젝트 안에 있어야 합니다.")
    if not directory.is_dir():
        raise RuntimeError(f"CSV 폴더가 없습니다: {directory}")
    files = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".csv" and not path.name.startswith("."):
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            files.append(path)
    files.sort(key=lambda path: path.name.casefold())
    if not files:
        raise RuntimeError(f"CSV 파일이 없습니다: {directory}")
    selected = files[0] if len(files) == 1 else files[choose("입력 CSV 선택", [p.name for p in files]) - 1]
    relative = selected.relative_to(root.resolve()).as_posix()
    print(f"→ 입력 CSV: {relative}")
    return relative


def result_directory(root, family, doc_dir):
    """선택한 자료의 연도·구분으로 겹치지 않는 결과 경로를 만든다."""
    dataset = Path(doc_dir).relative_to(Path(CONFIG["DOCUMENT_DIR"]))
    source, year = dataset.parts[:2]
    if source not in ("열린재정", "지방재정") or not year.isdigit():
        raise RuntimeError(f"결과 폴더에 사용할 자료 구분·연도를 확인할 수 없습니다: {doc_dir}")
    today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%d")
    parent = root / family / CONFIG["OUTPUT_SUBDIR"]
    prefix = f"{today}_{year}_{source}_"
    numbers = [int(path.name[len(prefix):]) for path in parent.iterdir()
               if path.name.startswith(prefix) and path.name[len(prefix):].isdigit()] if parent.is_dir() else []
    output = parent / f"{prefix}{max(numbers, default=0) + 1}"
    relative = output.relative_to(root).as_posix()
    print(f"→ 결과 저장 폴더: {relative}")
    return relative


def folder_name(prompt, default, parent):
    while True:
        name = ask(prompt, default)
        if name in (".", "..") or "/" in name or "\\" in name or "\x00" in name or Path(name).is_absolute():
            print("경로가 아닌 폴더 이름 하나만 입력해주세요.")
            continue
        if (parent / name).exists():
            print(f"이미 있는 폴더입니다: {parent / name}\n받아 둔 파일은 건너뛰고 이어서 수집합니다.")
            if ask("이 폴더를 사용할까요? (y/N)", "n").lower() != "y":
                continue
        return name


def has_key(crawler_dir, name):
    try:
        lines = (crawler_dir.parent / ".env").read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return False
    return any(line.split("=", 1)[0].strip() == name and line.split("=", 1)[1].strip()
               for line in lines if "=" in line and not line.lstrip().startswith("#"))


BROWSER_CHECK = ("from playwright.sync_api import sync_playwright\n"
                 "with sync_playwright() as p:\n"
                 "    p.chromium.launch().close()\n")


def crawl_command(root, crawler_dir):
    crawlers = (Path(crawler_dir).expanduser() if crawler_dir else root / CONFIG["CRAWLER_DIR"]).resolve()
    if not (crawlers / "openapi.py").is_file():
        raise RuntimeError(f"crawlers 폴더가 없습니다: {crawlers}")
    try:
        crawlers_path = crawlers.relative_to(root).as_posix()
    except ValueError:
        raise RuntimeError(f"crawlers 폴더는 프로젝트 폴더 안에 있어야 합니다: {root}")
    document = CONFIG["DOCUMENT_DIR"].rstrip("/")
    source = choose("수집 대상", ["지방재정365 (세부사업 목록 + 상세 PDF)",
                               "열린재정 (세부사업 목록 + 사업설명자료)",
                               "열린재정 Open API 예산 목록 (CSV만)"])
    year = integer("수집 회계연도", DEFAULTS["CRAWL_YEAR"] or datetime.date.today().year - 1, 2000)
    modules = []
    if source == 1:
        script, key = CONFIG["LOCAL_CRAWLER_SCRIPT"], "LOFIN365_API_KEY"
        available_regions = DEFAULTS["CRAWL_REGIONS"]
        regions = [available_regions[choose("수집 지역", available_regions) - 1]]
        date = DEFAULTS["CRAWL_DATE"] or f"{year}-12-31"
        if datetime.date.fromisoformat(date).year != int(year):
            raise RuntimeError("CRAWL_DATE의 연도와 선택한 수집 연도가 다릅니다.")
        list_only = DEFAULTS["CRAWL_LIST_ONLY"]
        limit = str(DEFAULTS["CRAWL_LIMIT"])
        name = dataset_name(year, regions[0])
        target = dataset_path(year, regions[0]).as_posix()
        options = ["--regions", *regions, "--date", date, "--limit", limit,
                   "--manifest", f"{target}/{name}.csv", "--output-dir", f"{target}/사업설명자료"]
        if not list_only:
            modules = ["playwright"]
    elif source == 2:
        script, key = CONFIG["OPEN_FISCAL_CRAWLER_SCRIPT"], None
        list_only = DEFAULTS["CRAWL_LIST_ONLY"]
        limit = str(DEFAULTS["CRAWL_LIMIT"])
        parent = f"{document}/열린재정"
        target = f"{parent}/{year}"
        options = ["--year", year, "--limit", limit, "--save-dir", parent]
        modules = ["pandas", "openpyxl", "tqdm", "playwright"]
    else:
        script, key = CONFIG["OPEN_FISCAL_API_SCRIPT"], "OPENFISCAL_API_KEY"
        list_only = False
        limit = str(DEFAULTS["CRAWL_API_LIMIT"])
        target = dataset_path(year).as_posix()
        options = ["--year", year, "--limit", limit, "--output", f"{target}/open_fiscal_api_{year}.csv"]
    if list_only:
        options.append("--list-only")
    if modules:
        # 컨테이너에는 Chrome이 없으므로 Playwright에 포함된 Chromium을 쓴다.
        options += ["--browser-channel", ""]
    if key and not has_key(crawlers, key):
        print(f"안내: {crawlers.parent / '.env'}에 {key}가 없습니다. 인증키 없이는 5건까지만 받을 수 있습니다.")
    if (root / target).exists():
        print("기존 수집 폴더를 사용합니다. 기존 파일 처리와 이어받기는 크롤러 규칙을 따릅니다.")
    container = str(CONFIG["DOCKER_CONTAINER"])
    workdir = str(CONFIG["DOCKER_WORKDIR"])
    print(f"저장 위치: {root / target}")
    print("안내: 호스트 프로젝트와 컨테이너 내부 프로젝트가 같은 폴더를 가리켜야 합니다.")
    cmd = ["docker", "exec", "-it", "-w", workdir, container, "python", "-u", f"{crawlers_path}/{script}", *options]
    return cmd, container, modules


def prepare_container(container, modules):
    """수집에 필요한 패키지와 브라우저가 컨테이너에 없으면 물어본 뒤 설치한다."""
    if not modules:
        return True
    check = "".join(f"import {module}\n" for module in modules) + BROWSER_CHECK
    quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if subprocess.run(["docker", "exec", container, "python", "-c", check], **quiet).returncode == 0:
        return True
    print(f"\n컨테이너({container})에 수집에 필요한 프로그램이 없습니다: {', '.join(modules)}, Chromium")
    if ask("지금 설치할까요? 몇 분 걸립니다. (y/N)", "n").lower() != "y":
        print("설치하지 않아 수집을 시작하지 않았습니다.")
        return False
    for step in (["python", "-m", "pip", "install", *modules],
                 ["python", "-m", "playwright", "install", "--with-deps", "chromium"]):
        if subprocess.run(["docker", "exec", container, *step]).returncode != 0:
            print("설치에 실패했습니다. 위 오류 내용을 관리자에게 전달해주세요.")
            return False
    print("설치를 마쳤습니다.")
    return True


def run_command(cmd, root, dry_run, prepare=None):
    print("\n실행할 명령어:\n" + shlex.join(cmd))
    if dry_run:
        print("미리보기 완료: 실행하지 않았습니다.")
        return 0
    if ask("실행할까요? (y/N)", "n").lower() != "y":
        print("취소했습니다.")
        return 0
    if not shutil.which("docker"):
        raise RuntimeError("docker 명령어를 찾을 수 없습니다.")
    if prepare and not prepare():
        return 1
    result = subprocess.run(cmd, cwd=root)
    print("\n완료했습니다." if result.returncode == 0 else f"\n실패했습니다. 종료 코드: {result.returncode}")
    return result.returncode


def run_bar(root, classified, amount_columns, unit, dry_run, docker=None):
    """분류 결과 CSV에 귀속률(BAR)을 적용한다. 추가 패키지가 필요 없다.

    docker=(컨테이너, 내부 프로젝트 경로)이면 분류를 실행한 컨테이너 안에서, 아니면 호스트 Python으로 실행한다.
    """
    options = ["--amount-cols", amount_columns, "--unit", unit]
    if docker:
        container, workdir = docker
        cmd = ["docker", "exec", "-w", workdir, container, "python", CONFIG["BAR_SCRIPT"],
               "--input", classified, "--rates", CONFIG["BAR_RATES"], *options]
    else:
        cmd = [sys.executable, str(root / CONFIG["BAR_SCRIPT"]), "--input", str(root / classified),
               "--rates", str(root / CONFIG["BAR_RATES"]), *options]
    if dry_run:
        print("\n이어서 실행할 귀속 예산(BAR) 계산 명령어:\n" + shlex.join(cmd))
        return 0
    print("\n귀속 예산(BAR) 계산")
    if not (root / classified).is_file():
        print(f"분류 결과 파일이 없어 귀속 예산 계산을 건너뜁니다: {root / classified}")
        return 0
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        print(f"귀속 예산 계산에 실패했습니다. 분류 결과는 그대로 있습니다. 종료 코드: {result.returncode}")
    return result.returncode


def is_final_result(path):
    """파이프라인이 끝난 뒤 남길 파일: 귀속 예산 결과(*_bar.csv)와 히트맵."""
    return path.name.endswith("_bar.csv") or "heatmap" in path.name.lower()


def keep_final_results(folder, since, dry_run):
    """이번 실행에서 만들어진 중간 파일(요약·캐시·검수·평가 등)을 지우고 최종 결과만 남긴다.

    실행 전부터 폴더에 있던 파일은 건드리지 않는다.
    """
    if dry_run:
        print(f"\n완료 후 {folder} 에는 *_bar.csv와 히트맵 파일만 남기고 이번 실행의 중간 파일은 삭제합니다.")
        return
    if not folder.is_dir() or not any(path.name.endswith("_bar.csv") for path in folder.iterdir()):
        print("\n귀속 예산 결과(*_bar.csv)가 없어 중간 파일을 지우지 않고 그대로 둡니다.")
        return
    removed = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and not is_final_result(path) and path.stat().st_mtime >= since - 1:
            path.unlink()
            removed.append(path.name)
    kept = [path.name for path in sorted(folder.iterdir()) if path.is_file() and is_final_result(path)]
    print(f"\n결과 정리: 중간 파일 {len(removed)}개 삭제, 남은 결과: {', '.join(kept) or '(없음)'}")


def restore_owner(root, container, workdir, output, dry_run):
    """docker exec는 컨테이너 root로 실행되므로 결과 폴더 소유자를 호스트 계정으로 되돌린다."""
    if not hasattr(os, "getuid"):
        return 0
    cmd = ["docker", "exec", container, "chown", "-R", f"{os.getuid()}:{os.getgid()}", f"{workdir}/{output}"]
    if dry_run:
        print("\n결과 폴더 소유자를 현재 계정으로 바꾸는 명령어:\n" + shlex.join(cmd))
        return 0
    if not (root / output).exists():
        return 0
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        print(f"결과 폴더 소유자를 바꾸지 못했습니다. 결과 파일은 그대로 있습니다. 종료 코드: {result.returncode}")
    return result.returncode


def augmentation_command(root):
    script = "transformer/v1/augmentation_h200/balanced_csv.py"
    if not (root / script).is_file():
        raise RuntimeError(f"증강 스크립트가 없습니다: {root / script}")
    if not hasattr(os, "getuid"):
        raise RuntimeError("증강 Docker 실행은 Linux 회사 서버에서 사용해주세요.")
    input_file = project_path("라벨이 있는 원본 CSV", CONFIG["DOCUMENT_DIR"], root, "file")
    output = project_path("증강 결과 폴더", "transformer/v1/outputs/augmentation", root, "output")
    backend = DEFAULTS["AUGMENTATION_BACKEND"]
    url = CONFIG["VLLM_URL" if backend == "vllm" else "OLLAMA_URL"]
    model = CONFIG["VLLM_MODEL" if backend == "vllm" else "OLLAMA_MODEL"]
    target = str(DEFAULTS["AUGMENTATION_TARGET"])
    mount = CONFIG["LLM_MOUNT_DIR"]
    cmd = ["docker", "run", "--rm", "-it", "--network", CONFIG["LLM_DOCKER_NETWORK"],
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "AUG_API_KEY"]
    if backend == "vllm":
        # Pass the host variable by name, never expose its value in the command.
        cmd += ["-e", CONFIG["VLLM_API_KEY_ENV"], "-e",
                f"AUG_API_KEY_ENV={CONFIG['VLLM_API_KEY_ENV']}"]
    cmd += ["-v", f"{root}:{mount}", "-w", mount, CONFIG["LLM_DOCKER_IMAGE"],
            "python", "-u", script, "--input", f"{mount}/{input_file}",
            "--output", f"{mount}/{output}", "--backend", backend,
            "--base-url", url, "--model", model, "--target-per-category", target,
            "--merged-output-dir", f"{mount}/{output}/merged_training"]
    print(f"원본+증강 합계가 가장 적은 1차 카테고리부터 50개씩 생성하고, 모든 카테고리가 {target}개가 되면 종료합니다. 같은 폴더로 재실행하면 이어서 생성합니다.")
    print("CSV 사업명·예산정보를 사용합니다. HWP/PDF 본문은 읽지 않습니다.")
    print(f"서버 저장 위치: {root / output / 'balanced_h200_v1'}")
    print(f"완료되면 학습용 통합 CSV를 자동 생성합니다: {root / output / 'merged_training' / 'open_fiscal_2023_augmented.csv'}")
    print("이미 증강이 끝난 폴더를 다시 선택하면 생성 없이 통합 CSV만 만듭니다.")
    return cmd


def validate_defaults():
    """명령어 생성 전에 실행 변수의 타입과 범위를 확인한다."""
    minimums = {"MAX_DOCUMENT_CHARS": 1, "MAX_TOKENS": 1, "WORKERS": 1,
                "NUM_CTX": 1, "TIMEOUT": 1, "RETRIES": 0, "TRAIN_EPOCHS": 1,
                "TRAIN_BATCH_SIZE": 1, "MAJORITY_CAP_MULTIPLIER": 1,
                "MAJORITY_CAP_MIN": 0, "CRAWL_LIMIT": 0, "CRAWL_API_LIMIT": 0,
                "AUGMENTATION_TARGET": 50}
    for key, minimum in minimums.items():
        value = DEFAULTS.get(key)
        if type(value) is not int or value < minimum:
            raise RuntimeError(f"defaults.json: {key}는 {minimum} 이상의 정수여야 합니다.")
    for key in ("UNDERSAMPLE_MAJORITY", "CLASS_WEIGHT", "CRAWL_LIST_ONLY"):
        if type(DEFAULTS.get(key)) is not bool:
            raise RuntimeError(f"defaults.json: {key}는 true 또는 false여야 합니다.")
    cap = DEFAULTS.get("TRAIN_AUGMENTATION_PER_CLASS")
    if cap is not None and (type(cap) is not int or cap < 0):
        raise RuntimeError("defaults.json: TRAIN_AUGMENTATION_PER_CLASS는 null 또는 0 이상 정수여야 합니다.")
    year = DEFAULTS.get("CRAWL_YEAR")
    if year is not None and (type(year) is not int or year < 2000):
        raise RuntimeError("defaults.json: CRAWL_YEAR는 null 또는 2000 이상 정수여야 합니다.")
    date = DEFAULTS.get("CRAWL_DATE")
    if date is not None:
        try:
            datetime.date.fromisoformat(date)
        except (ValueError, TypeError):
            raise RuntimeError("defaults.json: CRAWL_DATE는 null 또는 YYYY-MM-DD 날짜여야 합니다.")
    regions = DEFAULTS.get("CRAWL_REGIONS")
    if not isinstance(regions, list) or not regions or any(r not in REGIONS for r in regions):
        raise RuntimeError("defaults.json: CRAWL_REGIONS에 유효한 지역 목록을 지정해주세요.")
    if DEFAULTS.get("AUGMENTATION_BACKEND") not in ("vllm", "ollama"):
        raise RuntimeError("defaults.json: AUGMENTATION_BACKEND는 vllm 또는 ollama여야 합니다.")
    if DEFAULTS["AUGMENTATION_TARGET"] % 50:
        raise RuntimeError("defaults.json: AUGMENTATION_TARGET은 50의 배수여야 합니다.")
    if DEFAULTS.get("BAR_UNIT") not in ("원", "백만원"):
        raise RuntimeError("defaults.json: BAR_UNIT는 원 또는 백만원이어야 합니다.")
    rate = DEFAULTS.get("TRAIN_LEARNING_RATE")
    if type(rate) not in (int, float) or not 0 < rate < 1:
        raise RuntimeError("defaults.json: TRAIN_LEARNING_RATE는 0보다 크고 1보다 작아야 합니다.")


def main():
    validate_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=CONFIG["PROJECT_DIR"] or os.getcwd(), help="호스트의 biofin_cls3 프로젝트 경로")
    parser.add_argument("--crawler-dir", help="프로젝트 폴더 안의 crawlers 폴더 경로 (기본: crawlers)")
    parser.add_argument("--dry-run", action="store_true", help="명령어만 출력하고 실행하지 않음")
    args = parser.parse_args()
    root = Path(args.project_dir).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"프로젝트 폴더가 없습니다: {root}")
    print(f"BIOFIN 실행 도우미\n프로젝트: {root}\n파일과 폴더는 목록에서 번호로 선택하세요.")
    task = choose("작업 선택", ["모델 예측", "모델 학습", "크롤링", "데이터 증강"])
    if task == 4:
        return run_command(augmentation_command(root), root, args.dry_run)
    if task == 3:
        cmd, container, modules = crawl_command(root, args.crawler_dir)
        return run_command(cmd, root, args.dry_run, lambda: prepare_container(container, modules))
    training = task == 2
    backend = choose("모델 선택", ["Transformer", "LLM (Ollama)", "LLM (vLLM API)"])
    transformer = backend == 1
    vllm = backend == 3
    if not transformer:
        print("안내: LLM은 실행 중인 서버 API로 분류합니다. 모델 학습이나 서버 기동은 하지 않습니다.")
    vllm_script = CONFIG["VLLM_SCRIPT"]
    augmentation = None
    if transformer and training:
        augmentation = choose("학습 데이터 선택", ["원본만 사용", "원본 + 증강 통합 CSV 사용"])
    doc_dir = select_document_dir(root)
    csv_folder = CONFIG["AUGMENTATION_ARCHIVE_DIR"] if augmentation == 2 else Path(doc_dir).parent
    input_file = select_csv(root, csv_folder)
    if augmentation == 2:
        try:
            with (root / input_file).open(encoding="utf-8-sig", newline="") as stream:
                columns = set(next(csv.reader(stream), []))
        except UnicodeDecodeError:
            raise RuntimeError("증강 통합 CSV는 UTF-8 파일이어야 합니다.")
        if not {"row_type", "training_eligible"}.issubset(columns):
            raise RuntimeError("증강 학습에는 row_type, training_eligible 컬럼이 있는 통합 CSV가 필요합니다.")
    family = "transformer" if transformer else "llm"
    output = result_directory(root, family, doc_dir)
    if transformer:
        container = str(CONFIG["DOCKER_CONTAINER"])
        workdir = str(CONFIG["DOCKER_WORKDIR"])
        script = f"{CONFIG['TRANSFORMER_SCRIPT_DIR']}/{'train' if training else 'predict'}_attention_classifier.py"
        cmd = ["docker", "exec", "-it", "-w", workdir, container, "python", "-u", script]
        if training:
            cmd += ["--label_file", input_file, "--doc_dir", doc_dir,
                    "--epochs", str(DEFAULTS["TRAIN_EPOCHS"]),
                    "--batch_size", str(DEFAULTS["TRAIN_BATCH_SIZE"]),
                    "--learning_rate", str(DEFAULTS["TRAIN_LEARNING_RATE"])]
            if augmentation == 1:
                cmd += ["--augmentation_per_class", "0"]
            elif DEFAULTS["TRAIN_AUGMENTATION_PER_CLASS"] is not None:
                cmd += ["--augmentation_per_class", str(DEFAULTS["TRAIN_AUGMENTATION_PER_CLASS"])]
            if augmentation == 2:
                print("증강문은 학습용 원본 사업에 연결된 것만 사용하며 검증/시험 사업의 증강문은 제외됩니다.")
            if DEFAULTS["UNDERSAMPLE_MAJORITY"]:
                cmd += ["--undersample_majority", "--majority_label", str(DEFAULTS["MAJORITY_LABEL"]),
                        "--majority_cap_multiplier", str(DEFAULTS["MAJORITY_CAP_MULTIPLIER"]),
                        "--majority_cap_min", str(DEFAULTS["MAJORITY_CAP_MIN"])]
            if DEFAULTS["CLASS_WEIGHT"]:
                cmd += ["--class_weight"]
        else:
            model_path = (root / CONFIG["TRANSFORMER_MODEL_DIR"]).resolve()
            try:
                model_dir = model_path.relative_to(root).as_posix()
            except ValueError:
                raise RuntimeError("TRANSFORMER_MODEL_DIR는 프로젝트 안의 경로여야 합니다.")
            if not model_path.is_dir():
                raise RuntimeError(f"학습된 모델 폴더가 없습니다: {model_path}\nconfig.json의 TRANSFORMER_MODEL_DIR를 확인해주세요.")
            print(f"→ 학습된 모델 폴더: {model_dir}")
            cmd += ["--model_dir", model_dir, "--budget_file", input_file, "--doc_dir", doc_dir,
                    "--filename_column", str(DEFAULTS["FILENAME_COLUMN"])]
        cmd += ["--output_dir", output]
        print("안내: 호스트 프로젝트와 컨테이너 내부 프로젝트의 파일 구성이 같아야 합니다.")
    else:
        script = f"{CONFIG['LLM_SCRIPT_DIR']}/{vllm_script if vllm else 'classify_biofin_category_with_ollama.py'}"
        if not hasattr(os, "getuid"):
            raise RuntimeError("LLM Docker 실행은 Linux에서 사용해주세요.")
        cmd = ["docker", "run", "--rm", "-it", "--network", CONFIG["LLM_DOCKER_NETWORK"], "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", f"PYTHONPATH={CONFIG['LLM_PYTHONPATH']}"]
        if vllm:
            cmd += ["-e", CONFIG["VLLM_API_KEY_ENV"]]
        cmd += ["-v", f"{root}:{CONFIG['LLM_MOUNT_DIR']}", "-w", CONFIG["LLM_MOUNT_DIR"],
               CONFIG["LLM_DOCKER_IMAGE"], "python", "-u", script,
               "--input-file", f"{CONFIG['LLM_MOUNT_DIR']}/{input_file}", "--doc-dir", f"{CONFIG['LLM_MOUNT_DIR']}/{doc_dir}",
               "--output-dir", f"{CONFIG['LLM_MOUNT_DIR']}/{output}"]
        if vllm:
            cmd += ["--vllm-url", str(CONFIG["VLLM_URL"]),
                    "--model", str(CONFIG["VLLM_MODEL"])]
            print("최대 컨텍스트와 GPU 점유는 서버 설정입니다. 여기서는 응답 길이와 동시 요청 수를 정합니다.")
        else:
            cmd += ["--ollama-url", str(CONFIG["OLLAMA_URL"]),
                    "--model", str(CONFIG["OLLAMA_MODEL"])]
        cmd += ["--max-document-chars", str(DEFAULTS["MAX_DOCUMENT_CHARS"])]
        if vllm:
            cmd += ["--max-tokens", str(DEFAULTS["MAX_TOKENS"]),
                    "--workers", str(DEFAULTS["WORKERS"])]
        else:
            cmd += ["--num-ctx", str(DEFAULTS["NUM_CTX"])]
        cmd += ["--timeout", str(DEFAULTS["TIMEOUT"])]
        cmd += ["--retries", str(DEFAULTS["RETRIES"])]
    if not (root / script).is_file():
        raise RuntimeError(f"실행 스크립트가 없습니다: {root / script}")
    if training:
        code = run_command(cmd, root, args.dry_run)
        if transformer:
            return restore_owner(root, container, workdir, output, args.dry_run) or code
        return code
    # 분류가 오래 걸리므로 귀속 예산 설정은 실행 전에 묻는다.
    amount_columns = str(DEFAULTS["BAR_AMOUNT_COLUMNS"])
    unit = DEFAULTS["BAR_UNIT"]
    started = time.time()
    code = run_command(cmd, root, args.dry_run)
    if code == 0:
        classified = f"{output}/classified.csv" if transformer else f"{output}/{Path(input_file).stem}_llm_classified.csv"
        code = run_bar(root, classified, amount_columns, unit, args.dry_run,
                       (container, workdir) if transformer else None)
    if transformer:
        code = restore_owner(root, container, workdir, output, args.dry_run) or code
    if code == 0:
        keep_final_results(root / output, started, args.dry_run)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n취소했습니다.")
        raise SystemExit(130)
    except (RuntimeError, OSError) as error:
        print(f"오류: {error}")
        raise SystemExit(1)
