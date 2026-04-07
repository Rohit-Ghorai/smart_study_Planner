#!/usr/bin/env python
"""Quick test to verify UI updates rendered correctly."""

from study_planner.web import app

client = app.test_client()
response = client.get('/')

html = response.get_data(as_text=True)

print(f"Status: {response.status_code}")
print(f"Has hero stats: {'hero-stats' in html}")
print(f"Has actions grid: {'actions-grid' in html}")
print(f"Has '5 types': {'5 types' in html}")
print(f"Has action items: {'action-item' in html}")
print(f"Has Base Study: {'Base Study' in html}")
print(f"Has Deep Study: {'Deep Study' in html}")
print(f"Has Mock Tests: {'Mock Tests' in html}")
print(f"Has Breaks: {'Breaks' in html}")

if response.status_code == 200 and 'actions-grid' in html and '5 types' in html:
    print("\n✅ UI Update TEST PASSED")
else:
    print("\n❌ UI Update TEST FAILED")
