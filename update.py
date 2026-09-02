import re

with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

target = "    - 비용 충격 (WTI 유가): {f\"^{wt_s.iloc[-1]:.2f}^\" if not wt_s.empty else ^데이터 없음^}".replace("^", chr(39))
rep = """    - 비용 충격 (WTI 유가): {f\"^{wt_s.iloc[-1]:.2f}^\" if not wt_s.empty else ^데이터 없음^}
    - 장단기 금리차 (10년물-2년물): {f\"^{(b10_s.iloc[-1] - b2_s.iloc[-1]):.2f}%^\" if not b10_s.empty and not b2_s.empty else ^데이터 없음^}
    - 실물 경기 (Copper): {f\"^{cp_s.iloc[-1]:.2f}^\" if not cp_s.empty else ^데이터 없음^}
    - 물동량 (BDRY): {f\"^{fr_s.iloc[-1]:.2f}^\" if not fr_s.empty else ^데이터 없음^}
    - 달러 가치 (DXY): {f\"^{dx_s.iloc[-1]:.2f}^\" if not dx_s.empty else ^데이터 없음^}""".replace("^", chr(39))

if target in code:
    code = code.replace(target, rep)
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("Success")
else:
    print("Fail")
