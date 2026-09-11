#!/usr/bin/env python3
"""BIOFIN Docker command wizard (Linux, Python 3.8+; no dependencies)."""
import argparse
import datetime
import os
from pathlib import Path
import shlex
import shutil
import subprocess


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
    current = suggested.parent if kind in ("file", "output") else suggested
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=os.getcwd(), help="호스트의 biofin_cls3 프로젝트 경로")
    parser.add_argument("--dry-run", action="store_true", help="명령어만 출력하고 실행하지 않음")
    args = parser.parse_args()
    root = Path(args.project_dir).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"프로젝트 폴더가 없습니다: {root}")
    print(f"BIOFIN 실행 도우미\n프로젝트: {root}\n파일과 폴더는 목록에서 번호로 선택하세요.")
    training = choose("작업 선택", ["모델 학습", "모델 예측"]) == 1
    transformer = choose("모델 선택", ["Transformer", "LLM (Ollama)"]) == 1
    if not transformer:
        print("안내: 제공된 LLM 학습/예측은 모두 Ollama 분류 스크립트를 실행합니다.")
    data = "260812_2023data.csv" if training else "울산_환경부_2024/울산_환경부_2024.csv"
    docs = "2023/사업설명자료" if training else "울산_환경부_2024/사업설명자료"
    input_file = project_path("입력 CSV", f"document/{data}", root, "file")
    doc_dir = project_path("사업설명자료 폴더", f"document/{docs}", root, "dir")
    family = "transformer" if transformer else "llm"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = project_path("결과 저장 폴더", f"{family}/v1/outputs/{stamp}", root, "output")
    if transformer:
        container = ask("Docker 컨테이너 이름", "biofin")
        workdir = ask("컨테이너 내부 프로젝트 경로", "/work/biofin_cls3")
        script = f"transformer/v1/src/{'train' if training else 'predict'}_attention_classifier.py"
        cmd = ["docker", "exec", "-it", "-w", workdir, container, "python", "-u", script]
        if training:
            cmd += ["--label_file", input_file, "--doc_dir", doc_dir]
            if choose("다수 클래스 언더샘플링", ["사용", "사용하지 않음"]) == 1:
                cmd += ["--undersample_majority", "--majority_label", ask("다수 클래스 라벨", "0"),
                        "--majority_cap_multiplier", integer("다수 클래스 배수", 1),
                        "--majority_cap_min", integer("다수 클래스 최소 개수", 50, 0)]
            if choose("클래스 가중치", ["사용", "사용하지 않음"]) == 1:
                cmd += ["--class_weight"]
        else:
            model_dir = project_path("학습된 모델 폴더", "transformer/v1/outputs/260812_category_v1_zero50", root, "dir")
            cmd += ["--model_dir", model_dir, "--budget_file", input_file, "--doc_dir", doc_dir,
                    "--filename_column", ask("파일명 컬럼", "사업설명자료_파일명")]
        cmd += ["--output_dir", output]
        print("안내: 호스트 프로젝트와 컨테이너 내부 프로젝트의 파일 구성이 같아야 합니다.")
    else:
        script = "llm/v1/classify_biofin_category_with_ollama.py"
        if not hasattr(os, "getuid"):
            raise RuntimeError("LLM Docker 실행은 Linux에서 사용해주세요.")
        cmd = ["docker", "run", "--rm", "-it", "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
               "-e", "PYTHONPATH=/workspace/llm/.packages", "-v", f"{root}:/workspace", "-w", "/workspace",
               "python:3.11-slim", "python", "-u", script,
               "--input-file", f"/workspace/{input_file}", "--doc-dir", f"/workspace/{doc_dir}",
               "--output-dir", f"/workspace/{output}",
               "--ollama-url", ask("Ollama URL", "http://172.22.0.1:20001"),
               "--model", ask("Ollama 모델", "gemma3:12b"),
               "--max-document-chars", integer("문서 최대 문자 수", 6000),
               "--num-ctx", integer("컨텍스트 크기", 8192), "--timeout", integer("타임아웃(초)", 300)]
        retries = ask("재시도 횟수 (default: 스크립트 기본값)", "3" if training else "default")
        while retries != "default" and not retries.isdigit():
            retries = ask("0 이상 정수 또는 default", "default")
        if retries != "default":
            cmd += ["--retries", retries]
    if not (root / script).is_file():
        raise RuntimeError(f"실행 스크립트가 없습니다: {root / script}")
    print("\n실행할 명령어:\n" + shlex.join(cmd))
    if args.dry_run:
        print("미리보기 완료: 실행하지 않았습니다.")
        return 0
    if ask("실행할까요? (y/N)", "n").lower() != "y":
        print("취소했습니다.")
        return 0
    if not shutil.which("docker"):
        raise RuntimeError("docker 명령어를 찾을 수 없습니다.")
    result = subprocess.run(cmd, cwd=root)
    print("\n완료했습니다." if result.returncode == 0 else f"\n실패했습니다. 종료 코드: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n취소했습니다.")
        raise SystemExit(130)
    except (RuntimeError, OSError) as error:
        print(f"오류: {error}")
        raise SystemExit(1)
