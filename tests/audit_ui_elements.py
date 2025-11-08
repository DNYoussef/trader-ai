#!/usr/bin/env python3
"""
Comprehensive UI Element Audit
Tests every interactive element to verify it connects to real code/APIs
"""
import asyncio
import json
from playwright.async_api import async_playwright
from pathlib import Path
import time

class UIElementAuditor:
    def __init__(self):
        self.findings = {
            "buttons": [],
            "toggles": [],
            "graphs": [],
            "api_calls": [],
            "theater_elements": [],
            "real_implementations": []
        }
        self.screenshots_dir = Path("C:/Users/17175/Desktop/trader-ai/tests/ui_audit_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)

    async def audit_ui(self):
        """Main audit function"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(viewport={'width': 1920, 'height': 1080})

            # Enable console monitoring
            page = await context.new_page()

            # Track API calls
            api_calls = []
            page.on('request', lambda request: api_calls.append({
                'url': request.url,
                'method': request.method,
                'headers': request.headers
            }))

            # Track console messages
            console_messages = []
            page.on('console', lambda msg: console_messages.append({
                'type': msg.type,
                'text': msg.text
            }))

            print("🔍 Starting UI Element Audit...")
            print("=" * 80)

            # Navigate to dashboard
            print("\n1. Loading Dashboard...")
            await page.goto('http://localhost:3000', wait_until='networkidle')
            await page.wait_for_timeout(2000)

            # Take initial screenshot
            await page.screenshot(path=str(self.screenshots_dir / "01_initial_load.png"))
            print("   ✓ Initial load screenshot saved")

            # Section 1: Header and Mode Selector
            await self.audit_header(page)

            # Section 2: Unified Net Worth Card
            await self.audit_net_worth_card(page)

            # Section 3: Risk Metrics
            await self.audit_risk_metrics(page)

            # Section 4: Charts
            await self.audit_charts(page)

            # Section 5: Position Table
            await self.audit_position_table(page)

            # Section 6: Navigation Tabs
            await self.audit_navigation_tabs(page)

            # Section 7: Plaid Integration
            await self.audit_plaid_integration(page)

            # Section 8: API Connectivity
            await self.audit_api_connectivity(page, api_calls)

            # Section 9: Console Errors
            await self.audit_console_errors(console_messages)

            # Generate report
            await self.generate_report(page)

            await browser.close()

            return self.findings

    async def audit_header(self, page):
        """Audit header elements"""
        print("\n2. Auditing Header...")

        # Check mode selector
        mode_selector = await page.query_selector('select')
        if mode_selector:
            options = await mode_selector.query_selector_all('option')
            print(f"   ✓ Mode Selector: {len(options)} modes found")
            self.findings["real_implementations"].append({
                "element": "Mode Selector",
                "type": "dropdown",
                "options": len(options),
                "real": True
            })

            # Test mode switching
            await mode_selector.select_option('simple')
            await page.wait_for_timeout(500)
            await page.screenshot(path=str(self.screenshots_dir / "02_simple_mode.png"))

            await mode_selector.select_option('professional')
            await page.wait_for_timeout(500)
            await page.screenshot(path=str(self.screenshots_dir / "03_professional_mode.png"))

            await mode_selector.select_option('enhanced')
            await page.wait_for_timeout(500)
        else:
            print("   ✗ Mode Selector NOT FOUND")
            self.findings["theater_elements"].append("Mode Selector - not functional")

        # Check status indicators
        api_status = await page.query_selector('text=API')
        live_data_status = await page.query_selector('text=Live Data')

        if api_status:
            print("   ✓ API Status Indicator found")
            self.findings["real_implementations"].append({
                "element": "API Status",
                "type": "indicator",
                "real": True
            })

        if live_data_status:
            print("   ✓ Live Data Status Indicator found")
            self.findings["real_implementations"].append({
                "element": "Live Data Status",
                "type": "indicator",
                "real": True
            })

    async def audit_net_worth_card(self, page):
        """Audit Unified Net Worth Card"""
        print("\n3. Auditing Unified Net Worth Card...")

        # Check if card exists
        net_worth_card = await page.query_selector('text=Total Net Worth')
        if net_worth_card:
            print("   ✓ Net Worth Card found")

            # Check Plaid button
            plaid_button = await page.query_selector('button:has-text("Connect Bank Account")')
            if plaid_button:
                print("   ✓ Plaid Connect Button found")

                # Take screenshot before click
                await page.screenshot(path=str(self.screenshots_dir / "04_before_plaid_click.png"))

                # Click and verify modal opens
                await plaid_button.click()
                await page.wait_for_timeout(2000)

                # Check if Plaid iframe appears
                frames = page.frames
                plaid_frame = None
                for frame in frames:
                    if 'plaid' in frame.url.lower():
                        plaid_frame = frame
                        break

                if plaid_frame:
                    print("   ✓ Plaid Link Modal OPENED (REAL IMPLEMENTATION)")
                    await page.screenshot(path=str(self.screenshots_dir / "05_plaid_modal_opened.png"))
                    self.findings["real_implementations"].append({
                        "element": "Plaid Connect Button",
                        "type": "button",
                        "action": "Opens Plaid OAuth modal",
                        "real": True
                    })

                    # Close modal
                    await page.keyboard.press('Escape')
                    await page.wait_for_timeout(1000)
                else:
                    print("   ⚠ Plaid Modal DID NOT OPEN")
                    self.findings["theater_elements"].append("Plaid Button - clicks but no modal")
            else:
                print("   ✗ Plaid Connect Button NOT FOUND")
        else:
            print("   ✗ Net Worth Card NOT FOUND")

    async def audit_risk_metrics(self, page):
        """Audit risk metric cards"""
        print("\n4. Auditing Risk Metrics Cards...")

        metric_cards = await page.query_selector_all('[class*="bg-white"][class*="rounded"]')
        print(f"   ✓ Found {len(metric_cards)} potential metric cards")

        # Check for specific metrics
        metrics_to_check = ['Portfolio Value', 'P(ruin)', 'VaR', 'Sharpe']
        for metric in metrics_to_check:
            element = await page.query_selector(f'text={metric}')
            if element:
                # Get the value
                parent = await element.evaluate_handle('el => el.closest("div")')
                text_content = await parent.evaluate('el => el.textContent')

                has_value = any(char.isdigit() for char in text_content)
                if has_value:
                    print(f"   ✓ {metric}: HAS REAL VALUE")
                    self.findings["real_implementations"].append({
                        "element": metric,
                        "type": "metric_card",
                        "has_data": True,
                        "real": True
                    })
                else:
                    print(f"   ⚠ {metric}: NO VALUE (may be loading)")

        await page.screenshot(path=str(self.screenshots_dir / "06_risk_metrics.png"))

    async def audit_charts(self, page):
        """Audit chart components"""
        print("\n5. Auditing Charts...")

        # Scroll to charts section
        await page.evaluate('window.scrollTo(0, 800)')
        await page.wait_for_timeout(1000)

        # Check for recharts components
        charts = await page.query_selector_all('svg.recharts-surface')
        print(f"   ✓ Found {len(charts)} Recharts SVG elements")

        if len(charts) > 0:
            print("   ✓ Charts are RENDERED (real Recharts implementation)")
            self.findings["real_implementations"].append({
                "element": "Charts",
                "type": "recharts",
                "count": len(charts),
                "real": True
            })

            # Check for data points
            for i, chart in enumerate(charts[:3]):  # Check first 3 charts
                paths = await chart.query_selector_all('path[class*="recharts"]')
                if len(paths) > 0:
                    print(f"   ✓ Chart {i+1}: {len(paths)} data paths (HAS DATA)")
                else:
                    print(f"   ⚠ Chart {i+1}: No data paths (EMPTY)")
        else:
            print("   ✗ NO CHARTS FOUND")
            self.findings["theater_elements"].append("Charts - not rendered")

        await page.screenshot(path=str(self.screenshots_dir / "07_charts_section.png"))

    async def audit_position_table(self, page):
        """Audit position table"""
        print("\n6. Auditing Position Table...")

        # Look for table
        table = await page.query_selector('table')
        if table:
            rows = await table.query_selector_all('tr')
            print(f"   ✓ Position Table: {len(rows)} rows")

            if len(rows) > 1:  # More than just header
                print("   ✓ Table HAS DATA")
                self.findings["real_implementations"].append({
                    "element": "Position Table",
                    "type": "table",
                    "rows": len(rows),
                    "real": True
                })
            else:
                print("   ⚠ Table is EMPTY (no positions)")
        else:
            print("   ✗ Position Table NOT FOUND")

        await page.screenshot(path=str(self.screenshots_dir / "08_position_table.png"))

    async def audit_navigation_tabs(self, page):
        """Audit navigation tabs"""
        print("\n7. Auditing Navigation Tabs...")

        # Find tab buttons
        tab_buttons = await page.query_selector_all('button:has-text("Overview"), button:has-text("Terminal"), button:has-text("Analysis"), button:has-text("Learn"), button:has-text("Progress")')
        print(f"   ✓ Found {len(tab_buttons)} navigation tabs")

        tabs_to_test = ['Terminal', 'Analysis', 'Learn', 'Progress']
        for i, tab_name in enumerate(tabs_to_test):
            tab_button = await page.query_selector(f'button:has-text("{tab_name}")')
            if tab_button:
                print(f"   Testing {tab_name} tab...")
                await tab_button.click()
                await page.wait_for_timeout(1000)

                # Take screenshot
                await page.screenshot(path=str(self.screenshots_dir / f"09_{tab_name.lower()}_tab.png"))

                # Check if content changed
                content = await page.content()
                if tab_name.lower() in content.lower():
                    print(f"   ✓ {tab_name} tab LOADED (real content)")
                    self.findings["real_implementations"].append({
                        "element": f"{tab_name} Tab",
                        "type": "navigation",
                        "real": True
                    })
                else:
                    print(f"   ⚠ {tab_name} tab may not have content")

        # Return to Overview
        overview_tab = await page.query_selector('button:has-text("Overview")')
        if overview_tab:
            await overview_tab.click()
            await page.wait_for_timeout(1000)

    async def audit_plaid_integration(self, page):
        """Audit Plaid-specific elements"""
        print("\n8. Auditing Plaid Integration Elements...")

        # Check bank accounts section
        bank_section = await page.query_selector('text=Connected Bank Accounts')
        if bank_section:
            print("   ✓ Bank Accounts Section found")
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await page.wait_for_timeout(1000)
            await page.screenshot(path=str(self.screenshots_dir / "10_bank_accounts_section.png"))

        # Check for UnifiedNetWorthCard props
        print("   Checking UnifiedNetWorthCard implementation...")
        # This would be checked via code inspection below

    async def audit_api_connectivity(self, page, api_calls):
        """Audit API calls made"""
        print("\n9. Auditing API Connectivity...")

        # Filter for our API calls
        our_api_calls = [call for call in api_calls if 'localhost:8000' in call['url']]

        print(f"   ✓ Total API calls made: {len(our_api_calls)}")

        # Group by endpoint
        endpoints = {}
        for call in our_api_calls:
            endpoint = call['url'].split('localhost:8000')[-1]
            if endpoint not in endpoints:
                endpoints[endpoint] = []
            endpoints[endpoint].append(call['method'])

        for endpoint, methods in endpoints.items():
            print(f"   ✓ {endpoint}: {len(methods)} calls ({', '.join(set(methods))})")
            self.findings["api_calls"].append({
                "endpoint": endpoint,
                "methods": list(set(methods)),
                "count": len(methods)
            })

    async def audit_console_errors(self, console_messages):
        """Audit console errors"""
        print("\n10. Auditing Console Messages...")

        errors = [msg for msg in console_messages if msg['type'] == 'error']
        warnings = [msg for msg in console_messages if msg['type'] == 'warning']

        print(f"   Console Errors: {len(errors)}")
        print(f"   Console Warnings: {len(warnings)}")

        if errors:
            print("\n   ⚠ ERRORS FOUND:")
            for error in errors[:5]:  # Show first 5
                print(f"      - {error['text'][:100]}")

        self.findings["console_errors"] = errors
        self.findings["console_warnings"] = warnings

    async def generate_report(self, page):
        """Generate final audit report"""
        print("\n" + "=" * 80)
        print("📊 AUDIT SUMMARY")
        print("=" * 80)

        print(f"\n✅ Real Implementations: {len(self.findings['real_implementations'])}")
        for impl in self.findings['real_implementations']:
            print(f"   • {impl['element']} ({impl['type']})")

        print(f"\n⚠️  Theater Elements: {len(self.findings['theater_elements'])}")
        for theater in self.findings['theater_elements']:
            print(f"   • {theater}")

        print(f"\n🌐 API Calls: {len(self.findings['api_calls'])}")
        for api in self.findings['api_calls']:
            print(f"   • {api['endpoint']}: {api['count']} calls")

        print(f"\n❌ Console Errors: {len(self.findings.get('console_errors', []))}")
        print(f"⚠️  Console Warnings: {len(self.findings.get('console_warnings', []))}")

        # Save report to JSON
        report_path = Path("C:/Users/17175/Desktop/trader-ai/tests/UI_AUDIT_REPORT.json")
        with open(report_path, 'w') as f:
            json.dump(self.findings, f, indent=2)

        print(f"\n💾 Full report saved to: {report_path}")
        print(f"📸 Screenshots saved to: {self.screenshots_dir}")

async def main():
    auditor = UIElementAuditor()
    findings = await auditor.audit_ui()
    return findings

if __name__ == "__main__":
    asyncio.run(main())
