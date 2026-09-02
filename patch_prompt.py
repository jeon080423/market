import sys
import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_pattern = r'    - 비용 충격 \(WTI 유가\): \{f\"\{wt_s\.iloc\[-1\]:\.2f\}\" if not wt_s\.empty else \'데이터 없음\'\}'
new_replacement = '''    - 비용 충격 (WTI 유가): {f\"{wt_s.iloc[-1]:.2f}\" if not wt_s.empty else '데이터 없음'}
    - 장단기 금리차 (10년-2년): {f\"{(b10_s.iloc[-1] - b2_s.iloc[-1]):.2f}%\" if not b10_s.empty and not b2_s.empty else '데이터 없음'}
    - 실물 경기 (Copper): {f\"{cp_s.iloc[-1]:.2f}\" if not cp_s.empty else '데이터 없음'}
    - 물동량 (BDRY): {f\"{fr_s.iloc[-1]:.2f}\" if not fr_s.empty else '데이터 없음'}
    - 달러 가치 (DXY): {f\"{dx_s.iloc[-1]:.2f}\" if not dx_s.empty else '데이터 없음'}'''

new_code = re.sub(old_pattern, new_replacement, code)
if code != new_code:
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_code)
    print('Successfully updated app.py')
else:
    print('Pattern not found!')
