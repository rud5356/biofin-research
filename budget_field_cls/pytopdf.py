"""
HTML 슬라이드를 PDF로 변환하는 유틸리티 스크립트.

Selenium(Chrome 브라우저 자동화)을 사용해 HTML 슬라이드 파일을 열고,
각 슬라이드를 PDF 페이지로 캡처한 뒤 하나의 PDF로 합칩니다.

필요 패키지:
    pip install selenium webdriver-manager pypdf

사용법:
    python pytopdf.py
    (파일 하단의 slides_to_pdf() 호출 경로를 수정하여 사용하세요)
"""

import base64
import io
import time
from pathlib import Path

from pypdf import PdfWriter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def slides_to_pdf(html_path: str, pdf_path: str, num_slides: int = 11) -> None:
    """
    HTML 슬라이드 파일을 PDF로 변환합니다.

    Args:
        html_path: 변환할 HTML 파일 경로
        pdf_path: 저장할 PDF 파일 경로
        num_slides: 슬라이드 총 개수 (기본값: 11)
    """
    # ─── Chrome 브라우저 옵션 설정 ─────────────────────────────────────────
    options = Options()
    options.add_argument("--headless=new")          # 화면 없이 백그라운드로 실행
    options.add_argument("--window-size=1280,720")  # 슬라이드 비율에 맞는 창 크기
    options.add_argument("--no-sandbox")            # 리눅스 환경에서 필요한 옵션
    options.add_argument("--disable-dev-shm-usage") # 메모리 부족 방지

    # ChromeDriver를 자동으로 다운로드하여 Chrome에 연결
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )

    # HTML 파일을 브라우저에서 엽니다 (file:/// 프로토콜 사용)
    driver.get(f"file:///{Path(html_path).absolute()}")

    # 모든 CSS 애니메이션을 비활성화합니다.
    # 애니메이션이 있으면 캡처 타이밍에 따라 슬라이드가 중간 상태로 찍힐 수 있습니다.
    driver.execute_script("""
        const style = document.createElement('style');
        style.textContent = `
            *, *::before, *::after {
                animation-duration: 0s !important;
                animation-delay: 0s !important;
                transition-duration: 0s !important;
                transition-delay: 0s !important;
            }
        `;
        document.head.appendChild(style);
    """)
    time.sleep(0.3)  # 스타일 적용 대기

    # ─── 슬라이드별 PDF 캡처 ───────────────────────────────────────────────
    pdf_pages: list[bytes] = []
    for slide_index in range(num_slides):
        # JavaScript로 해당 슬라이드만 표시하고 나머지는 숨깁니다.
        # slides[i]: HTML에서 .slide 클래스를 가진 요소들 중 i번째
        driver.execute_script(f"""
            const slides = document.querySelectorAll('.slide');
            slides.forEach((s, idx) => {{
                s.classList.remove('active', 'exit');
                s.style.opacity = '0';
                s.style.pointerEvents = 'none';
            }});
            const target = slides[{slide_index}];
            target.classList.add('active');
            target.style.opacity = '1';
            target.style.pointerEvents = 'all';
            target.style.transform = 'none';
        """)
        time.sleep(0.8)  # 슬라이드 전환 후 렌더링 완료를 기다립니다

        # Chrome DevTools Protocol(CDP)로 현재 페이지를 PDF로 변환
        # paperWidth/Height: 인치 단위 (10×7.8인치 = 16:9 비율)
        result = driver.execute_cdp_cmd("Page.printToPDF", {
            "paperWidth": 10,
            "paperHeight": 7.8,
            "printBackground": True,  # 배경색 포함
            "marginTop": 0,
            "marginBottom": 0,
            "marginLeft": 0,
            "marginRight": 0,
            "scale": 0.8,             # 슬라이드가 페이지에 꽉 차도록 축소
        })
        # CDP는 PDF를 base64로 인코딩해서 반환합니다 → 디코딩하여 바이트로 변환
        pdf_pages.append(base64.b64decode(result["data"]))
        print(f"  슬라이드 {slide_index + 1}/{num_slides} 캡처 완료")

    driver.quit()

    # ─── 개별 PDF 페이지들을 하나의 파일로 합치기 ────────────────────────
    # PdfWriter: pypdf 라이브러리의 PDF 병합 도구
    pdf_writer = PdfWriter()
    for page_bytes in pdf_pages:
        # io.BytesIO: 바이트를 파일처럼 다루는 메모리 버퍼
        pdf_writer.append(io.BytesIO(page_bytes))

    with open(pdf_path, "wb") as output_file:
        pdf_writer.write(output_file)

    print(f"\n저장 완료: {pdf_path}")


# ─── 실행 진입점 ──────────────────────────────────────────────────────────────
# 이 파일을 직접 실행할 때만 동작합니다 (다른 파일에서 import 시에는 무시)
if __name__ == "__main__":
    slides_to_pdf(
        html_path="C:/Yuna/BIOFIN_TEXT_CLS/budget-classification-slides.html",
        pdf_path="C:/Yuna/BIOFIN_TEXT_CLS/output.pdf",
    )
