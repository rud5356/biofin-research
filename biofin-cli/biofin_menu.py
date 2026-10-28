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


# 실행 기본값은 같은 폴더의 config.json에서 관리합니다.
CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config(path=CONFIG_PATH):
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"설정 파일을 읽을 수 없습니다: {path} ({exc})") from exc
    if not isinstance(config, dict):
        raise ValueError(f"설정 파일은 JSON 객체여야 합니다: {path}")
    return config


try:
    CONFIG = load_config()
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
        return relative.as_posix()


REGIONS = ["서울", "부산", "대구", "대전", "인천", "광주", "울산", "세종", "경기",
           "충북", "충남", "전남", "경북", "경남", "제주", "강원", "전북"]


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
    document = root / "document"
    source = choose("수집 대상", ["지방재정365 (세부사업 목록 + 상세 PDF)",
                               "열린재정 (세부사업 목록 + 사업설명자료)",
                               "열린재정 Open API 예산 목록 (CSV만)"])
    year = integer("회계연도", datetime.date.today().year - 1, 2000)
    modules = []
    if source == 1:
        script, key = CONFIG["LOCAL_CRAWLER_SCRIPT"], "LOFIN365_API_KEY"
        regions = REGIONS
        if choose("지역 선택", ["전국 17개 시도", "직접 입력"]) == 2:
            while True:
                regions = ask("지역 이름 (공백으로 구분, 예: 강원 세종)").split()
                unknown = [region for region in regions if region not in REGIONS]
                if not unknown:
                    break
                print(f"알 수 없는 지역: {' '.join(unknown)}\n선택 가능: {' '.join(REGIONS)}")
        date = ask("기준일 (YYYY-MM-DD)", f"{year}-12-31")
        list_only = choose("수집 범위", ["목록만", "목록 + 상세 PDF"]) == 1
        limit = integer("지역별 최대 사업 수 (0: 전체)", 5, 0)
        name = folder_name("저장 폴더 이름", f"{regions[0] if len(regions) == 1 else '지방재정'}_{year}", document)
        target = f"document/{name}"
        options = ["--regions", *regions, "--date", date, "--limit", limit,
                   "--manifest", f"{target}/{name}.csv", "--output-dir", f"{target}/사업설명자료"]
        if not list_only:
            modules = ["playwright"]
    elif source == 2:
        script, key = CONFIG["OPEN_FISCAL_CRAWLER_SCRIPT"], None
        list_only = choose("수집 범위", ["목록만", "목록 + 사업설명자료"]) == 1
        limit = integer("최대 사업 수 (0: 전체)", 5, 0)
        print(f"열린재정은 <상위 폴더>/{year}/ 아래에 저장합니다. 상위 폴더가 필요 없으면 -를 입력하세요.")
        name = folder_name("상위 폴더 이름", "-", document)
        parent = "document" if name == "-" else f"document/{name}"
        target = f"{parent}/{year}"
        options = ["--year", year, "--limit", limit, "--save-dir", parent]
        modules = ["pandas", "openpyxl", "tqdm", "playwright"]
    else:
        script, key = CONFIG["OPEN_FISCAL_API_SCRIPT"], "OPENFISCAL_API_KEY"
        list_only = False
        limit = integer("최대 사업 수 (0: 전체)", 0, 0)
        name = folder_name("저장 폴더 이름", f"열린재정_API_{year}", document)
        target = f"document/{name}"
        options = ["--year", year, "--limit", limit, "--output", f"{target}/{name}.csv"]
    if list_only:
        options.append("--list-only")
    if modules:
        # 컨테이너에는 Chrome이 없으므로 Playwright에 포함된 Chromium을 쓴다.
        options += ["--browser-channel", ""]
    if key and not has_key(crawlers, key):
        print(f"안내: {crawlers.parent / '.env'}에 {key}가 없습니다. 인증키 없이는 5건까지만 받을 수 있습니다.")
    container = ask("Docker 컨테이너 이름", CONFIG["DOCKER_CONTAINER"])
    workdir = ask("컨테이너 내부 프로젝트 경로", CONFIG["DOCKER_WORKDIR"])
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



def augmentation_command(root):
    script = "transformer/v1/augmentation_h200/balanced_csv.py"
    if not (root / script).is_file():
        raise RuntimeError(f"증강 스크립트가 없습니다: {root / script}")
    if not hasattr(os, "getuid"):
        raise RuntimeError("증강 Docker 실행은 Linux 회사 서버에서 사용해주세요.")
    input_file = project_path("라벨이 있는 원본 CSV", CONFIG["DOCUMENT_DIR"], root, "file")
    output = project_path("증강 결과 폴더", "transformer/v1/outputs/augmentation", root, "output")
    backend = "vllm" if choose("증강 API", ["vLLM", "Ollama"]) == 1 else "ollama"
    url = ask("API 기본 URL (/v1 제외)", CONFIG["VLLM_URL" if backend == "vllm" else "OLLAMA_URL"])
    model = ask("모델 ID", CONFIG["VLLM_MODEL" if backend == "vllm" else "OLLAMA_MODEL"])
    while True:
        target = integer("카테고리별 최대 건수 (원본+증강, 50의 배수)", 400, 50)
        if int(target) % 50 == 0:
            break
        print("50의 배수를 입력해주세요.")
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
            "--base-url", url, "--model", model, "--target-per-category", target]
    print(f"원본+증강 합계가 가장 적은 1차 카테고리부터 50개씩 생성하고, 모든 카테고리가 {target}개가 되면 종료합니다. 같은 폴더로 재실행하면 이어서 생성합니다.")
    print("CSV 사업명·예산정보를 사용합니다. HWP/PDF 본문은 읽지 않습니다.")
    print("결과는 balanced_h200_v1/의 JSONL입니다. 학습 전 build_merged_training_csv.py로 병합하세요.")
    print(f"서버 저장 위치: {root / output / 'balanced_h200_v1'}")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=CONFIG["PROJECT_DIR"] or os.getcwd(), help="호스트의 biofin_cls3 프로젝트 경로")
    parser.add_argument("--crawler-dir", help="프로젝트 폴더 안의 crawlers 폴더 경로 (기본: crawlers)")
    parser.add_argument("--dry-run", action="store_true", help="명령어만 출력하고 실행하지 않음")
    args = parser.parse_args()
    root = Path(args.project_dir).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"프로젝트 폴더가 없습니다: {root}")
    print(f"BIOFIN 실행 도우미\n프로젝트: {root}\n파일과 폴더는 목록에서 번호로 선택하세요.")
    task = choose("작업 선택", ["모델 학습", "모델 예측", "자료 수집 (크롤러)", "데이터 증강 (카테고리별 최대 400개)"])
    if task == 4:
        return run_command(augmentation_command(root), root, args.dry_run)
    if task == 3:
        cmd, container, modules = crawl_command(root, args.crawler_dir)
        return run_command(cmd, root, args.dry_run, lambda: prepare_container(container, modules))
    training = task == 1
    backend = choose("모델 선택", ["Transformer", "LLM (Ollama)", "LLM (vLLM API)"])
    transformer = backend == 1
    vllm = backend == 3
    if not transformer:
        print("안내: LLM은 실행 중인 서버 API로 분류합니다. 모델 학습이나 서버 기동은 하지 않습니다.")
    version_suffix = ""
    if vllm:
        versions = CONFIG["VLLM_SCRIPT_VERSIONS"]
        _, vllm_script, version_suffix = versions[choose("분류 프롬프트 버전", [label for label, _, _ in versions]) - 1]
    augmentation = None
    if transformer and training:
        augmentation = choose("학습 데이터 선택", ["원본만 사용", "원본 + 증강 통합 CSV 사용"])
    while True:
        input_file = project_path("원본 + 증강 통합 CSV" if augmentation == 2 else "입력 CSV", CONFIG["DOCUMENT_DIR"], root, "file")
        if augmentation != 2:
            break
        try:
            with (root / input_file).open(encoding="utf-8-sig", newline="") as stream:
                columns = set(next(csv.reader(stream), []))
            if {"row_type", "training_eligible"}.issubset(columns):
                break
        except UnicodeDecodeError:
            pass
        print("증강 학습에는 row_type, training_eligible 컬럼이 있는 UTF-8 통합 CSV를 선택해주세요.")
    doc_dir = project_path("사업설명자료 폴더", CONFIG["DOCUMENT_DIR"], root, "dir")
    family = "transformer" if transformer else "llm"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + version_suffix
    output = project_path("결과 저장 폴더", f"{family}/{CONFIG['OUTPUT_SUBDIR']}/{stamp}", root, "output")
    if transformer:
        container = ask("Docker 컨테이너 이름", CONFIG["DOCKER_CONTAINER"])
        workdir = ask("컨테이너 내부 프로젝트 경로", CONFIG["DOCKER_WORKDIR"])
        script = f"{CONFIG['TRANSFORMER_SCRIPT_DIR']}/{'train' if training else 'predict'}_attention_classifier.py"
        cmd = ["docker", "exec", "-it", "-w", workdir, container, "python", "-u", script]
        if training:
            cmd += ["--label_file", input_file, "--doc_dir", doc_dir]
            if augmentation == 1:
                cmd += ["--augmentation_per_class", "0"]
            elif choose("증강 데이터 사용량", ["사용 가능한 증강 전체", "클래스별 최대 개수 지정"]) == 2:
                cmd += ["--augmentation_per_class", integer("클래스별 증강 최대 개수", 100)]
            if augmentation == 2:
                print("증강문은 학습용 원본 사업에 연결된 것만 사용하며 검증/시험 사업의 증강문은 제외됩니다.")
            if choose("다수 클래스 언더샘플링", ["사용", "사용하지 않음"]) == 1:
                cmd += ["--undersample_majority", "--majority_label", ask("다수 클래스 라벨", "0"),
                        "--majority_cap_multiplier", integer("다수 클래스 배수", 1),
                        "--majority_cap_min", integer("다수 클래스 최소 개수", 50, 0)]
            if choose("클래스 가중치", ["사용", "사용하지 않음"]) == 1:
                cmd += ["--class_weight"]
        else:
            model_dir = project_path("학습된 모델 폴더", CONFIG["TRANSFORMER_MODEL_DIR"], root, "dir")
            cmd += ["--model_dir", model_dir, "--budget_file", input_file, "--doc_dir", doc_dir,
                    "--filename_column", ask("파일명 컬럼", "사업설명자료_파일명")]
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
            cmd += ["--vllm-url", ask("vLLM API URL", CONFIG["VLLM_URL"]),
                    "--model", ask("vLLM 모델 ID (/v1/models의 id)", CONFIG["VLLM_MODEL"])]
            print("최대 컨텍스트와 GPU 점유는 서버 설정입니다. 여기서는 응답 길이와 동시 요청 수를 정합니다.")
        else:
            cmd += ["--ollama-url", ask("Ollama URL", CONFIG["OLLAMA_URL"]),
                    "--model", ask("Ollama 모델", CONFIG["OLLAMA_MODEL"])]
        cmd += ["--max-document-chars", integer("사업목적 최대 문자 수", CONFIG["MAX_DOCUMENT_CHARS"])]
        if vllm:
            cmd += ["--max-tokens", integer("응답 최대 토큰 수", CONFIG["MAX_TOKENS"]),
                    "--workers", integer("동시 API 요청 수", CONFIG["WORKERS"])]
        else:
            cmd += ["--num-ctx", integer("컨텍스트 크기", CONFIG["NUM_CTX"])]
        cmd += ["--timeout", integer("타임아웃(초)", CONFIG["TIMEOUT"])]
        retries = ask("재시도 횟수 (default: 스크립트 기본값)", "3" if training else "default")
        while retries != "default" and not retries.isdigit():
            retries = ask("0 이상 정수 또는 default", "default")
        if retries != "default":
            cmd += ["--retries", retries]
    if not (root / script).is_file():
        raise RuntimeError(f"실행 스크립트가 없습니다: {root / script}")
    return run_command(cmd, root, args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n취소했습니다.")
        raise SystemExit(130)
    except (RuntimeError, OSError) as error:
        print(f"오류: {error}")
        raise SystemExit(1)
