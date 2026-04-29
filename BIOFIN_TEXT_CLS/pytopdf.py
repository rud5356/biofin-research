from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
import base64, time, io
from pypdf import PdfWriter

def slides_to_pdf(html_path, pdf_path, num_slides=11):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1280,720")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    driver.get(f"file:///{Path(html_path).absolute()}")
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
    time.sleep(0.3)
    
    pdfs = []
    for i in range(num_slides):
        # JS로 직접 해당 슬라이드로 이동
        driver.execute_script(f"""
            const slides = document.querySelectorAll('.slide');
            slides.forEach((s, idx) => {{
                s.classList.remove('active', 'exit');
                s.style.opacity = '0';
                s.style.pointerEvents = 'none';
            }});
            const target = slides[{i}];
            target.classList.add('active');
            target.style.opacity = '1';
            target.style.pointerEvents = 'all';
            target.style.transform = 'none';
        """)
        time.sleep(0.8)  # 0.6 → 0.8

        result = driver.execute_cdp_cmd("Page.printToPDF", {
            "paperWidth": 10,
            "paperHeight": 7.8,
            "printBackground": True,
            "marginTop": 0,
            "marginBottom": 0,
            "marginLeft": 0,
            "marginRight": 0,
            "scale": 0.8, 
        })
        pdfs.append(base64.b64decode(result["data"]))
        print(f"  슬라이드 {i+1}/{num_slides} 완료")

    driver.quit()

    writer = PdfWriter()
    for pdf_bytes in pdfs:
        writer.append(io.BytesIO(pdf_bytes))
    with open(pdf_path, "wb") as f:
        writer.write(f)
    print(f"\n저장 완료: {pdf_path}")

slides_to_pdf("C:/Yuna/BIOFIN_TEXT_CLS/budget-classification-slides.html", "C:/Yuna/BIOFIN_TEXT_CLS/output.pdf")