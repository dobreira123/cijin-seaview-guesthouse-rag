"""Tool-choice test: does the agent pick the right tool (database, calculation, RAG, booking, hand-over)?

For each question we list the tool(s) we accept. "none" means the agent should answer without any tool
(a polite refusal). Write tools are never confirmed here, so the test does not change any booking.
Run it with:  python test_tool_choice.py        (add a number, e.g. 5, to wait between questions)
"""
import sys
import time

from app import run_agent

TESTS = [
    # (question, accepted tools)
    ("What time is check-in?",                                         {"get_guesthouse_info"}),
    ("請問有停車場嗎？",                                                  {"get_guesthouse_info"}),
    ("Do you have a room for 6 people?",                               {"search_rooms"}),
    ("海景雙人房一晚多少錢？",                                              {"search_rooms", "calculate_stay"}),
    ("How much are 3 nights in the Twin Room with 2 breakfasts each day?", {"calculate_stay"}),
    ("If a typhoon comes on my check-in day, can I get my deposit back?", {"search_house_rules"}),
    ("My grandmother uses a wheelchair. Is there an elevator?",        {"search_house_rules"}),
    ("可以在房間抽電子菸嗎？",                                              {"search_house_rules"}),
    ("Can my 5-year-old son sleep in our bed for free?",               {"search_house_rules"}),
    ("退房時忘了東西，你們會保留多久？",                                       {"search_house_rules"}),
    ("Please check my booking B1003, phone ends with 0917.",           {"get_booking"}),
    ("我要取消訂房 B1009，電話末四碼 1290",                                  {"cancel_booking"}),
    ("I want to book the Twin Room from 2026-12-10 for 2 nights, 2 people, name Wang Xiao-Ming, phone 0912345678.",
                                                                       {"make_booking"}),
    ("The air conditioner was broken and I want a refund.",            {"transfer_to_human"}),
    ("What is the weather in Kaohsiung tomorrow?",                     {"none"}),
    ("幫我寫一個 Python 的排序程式",                                         {"none"}),
]

SLEEP_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    passed = 0
    print(f"{'#':>2}  {'result':6}  {'tools called':34}  question")
    print("-" * 100)
    for i, (question, accepted) in enumerate(TESTS, start=1):
        _, trace = run_agent([{"role": "user", "content": question}])
        called = [t["tool"] for t in trace] or ["none"]
        ok = called[0] in accepted
        passed += ok
        print(f"{i:>2}  {'PASS' if ok else 'MISS':6}  {', '.join(called)[:34]:34}  {question[:60]}"
              + ("" if ok else f"   (accepted: {', '.join(sorted(accepted))})"))
        time.sleep(SLEEP_SECONDS)
    print("-" * 100)
    print(f"Score: {passed}/{len(TESTS)} = {passed / len(TESTS):.0%}")


if __name__ == "__main__":
    main()
