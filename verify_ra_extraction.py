import asyncio
from playwright.async_api import async_playwright

HTML_CONTENT = """
<html>
<body>
<div class="card">
    <div class="block_header">
        <p class="bid_no pull-left">
          <span class="bid_title">RA NO:&nbsp;</span>
          <a class="bid_no_hover"
             href="/showradocumentPdf/8769074"
             target="_blank">
             GEM/2025/R/597611
          </a>
        </p>
    </div>
</div>
</body>
</html>
"""

async def verify():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(HTML_CONTENT)
        
        c = await page.query_selector("div.card")
        
        # Test the selector logic
        ra_no = ""
        ra_url = ""
        BASE_URL = "https://bidplus.gem.gov.in"
        
        try:
            ra_p = await c.query_selector("p.bid_no:has(span.bid_title:text('RA NO:'))")
            if ra_p:
                ra_link = await ra_p.query_selector("a")
                if ra_link:
                    ra_no = (await ra_link.inner_text()).strip()
                    href = await ra_link.get_attribute("href")
                    if href:
                        if href.startswith("/"):
                            ra_url = BASE_URL + href
                        else:
                            ra_url = href
        except Exception as e:
            print(f"Error: {e}")

        print(f"Extracted RA NO: '{ra_no}'")
        print(f"Extracted RA URL: '{ra_url}'")
        
        # Assertion
        assert ra_no == "GEM/2025/R/597611"
        assert ra_url == "https://bidplus.gem.gov.in/showradocumentPdf/8769074"
        print("✅ Verification Passed!")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(verify())
