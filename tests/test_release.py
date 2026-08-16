

def test_승인문서에_사용자_경로가_안_찍힌다():
    """**승인 요청 문서는 시연에서 화면에 띄우는 자리다.**

    실측에서 `C:\\Users\\<사번>\\Desktop\\...` 이 그대로 찍혀 나갔다.
    `scripts/check_public.py` 는 추적 파일만 보므로 **실행 결과로 만들어지는
    이 문서는 못 본다.** 그래서 여기서 잡는다.
    """
    from agents.release import display_path

    # **시험 자료에 진짜 사용자명을 쓰지 않는다.** 처음에 실제 사번을 그대로
    # 넣었는데, 사용자 경로가 새 나가는 것을 막으려는 시험이 그 사번을
    # 저장소에 커밋해 버렸다. 4090 에서 `check_public.py` 가 잡았다.
    # 무엇으로 재느냐는 아무 문자열이나 되고, 진짜 값일 이유가 없다.
    leaky = [
        "/Users/someuser/Desktop/Project/SK/release/pcb1-01-v2",
        "/home/otheruser/work/release/pcb1-01-v2",
        "C:\\Users\\someuser\\Desktop\\SK\\release\\pcb1-01-v2",
    ]
    for raw in leaky:
        shown = display_path(raw)
        low = shown.lower().replace("\\", "/")
        assert "users/" not in low and "home/" not in low, f"{raw} → {shown}"
        assert "someuser" not in low and "otheruser" not in low, f"{raw} → {shown}"
        # 승인자가 어느 폴더인지는 알아야 한다.
        assert "pcb1-01-v2" in shown, f"{raw} → {shown}"
