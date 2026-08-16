

def test_승인문서에_사용자_경로가_안_찍힌다():
    """**승인 요청 문서는 시연에서 화면에 띄우는 자리다.**

    실측에서 `C:\\Users\\<사번>\\Desktop\\...` 이 그대로 찍혀 나갔다.
    `scripts/check_public.py` 는 추적 파일만 보므로 **실행 결과로 만들어지는
    이 문서는 못 본다.** 그래서 여기서 잡는다.
    """
    from agents.release import display_path

    leaky = [
        "/Users/so23696/Desktop/Project/SK/release/pcb1-01-v2",
        "/home/gildong/work/release/pcb1-01-v2",
        "C:\\Users\\so23696\\Desktop\\SK\\release\\pcb1-01-v2",
    ]
    for raw in leaky:
        shown = display_path(raw)
        low = shown.lower().replace("\\", "/")
        assert "users/" not in low and "home/" not in low, f"{raw} → {shown}"
        assert "so23696" not in low and "gildong" not in low, f"{raw} → {shown}"
        # 승인자가 어느 폴더인지는 알아야 한다.
        assert "pcb1-01-v2" in shown, f"{raw} → {shown}"
