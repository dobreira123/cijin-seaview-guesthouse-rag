# Xiao Hai (小海) 2.0: an Agent + RAG Help Desk for Cijin Seaview Guesthouse

Xiao Hai is an AI help desk for **Cijin Seaview Guesthouse (旗津海景民宿)**, a small, made-up guesthouse near the
Cijin Ferry Pier in Kaohsiung. Guests can ask about rooms and prices, ask about the long house rules (typhoons,
children, pets, stairs, smoking ...), look up their booking, book a room, or cancel a booking, in Chinese or English,
at any time of the day.

The answers do not come from the model's memory:

- **short facts and prices** come from a SQLite database through tools,
- **long rules** come from a **RAG** search over `house_rules.md`,
- **bookings** are read from and written to the database by tools, only after the guest says yes,
- the agent **calls the tools** in a loop and shows each call in a trace panel,
- anything it cannot answer goes to the front desk.

> Midterm project, Deep Learning (NKUST): Your Midterm Project: Agent + RAG + LLM Application
> Student: F115119131 葉日杰. It grows out of my Unit G homework
> ([cijin-seaview-guesthouse-agent](https://github.com/dobreira123/cijin-seaview-guesthouse-agent)).

![Main screen: a RAG answer with the tool trace](images/screenshot-rag.png)

## Who It Serves and Why

- **Users:** travellers who plan a trip to Cijin Island, and guests who already booked a room.
- **When:** in the evening, when they plan the trip, and when the one-person front desk (09:00-21:00) is busy.
- **What they need:** exact prices, clear answers to "what if" questions (a typhoon, a child, a wheelchair, a pet),
  and a quick way to check or cancel a booking.
- **Why a general AI chat cannot do this:** it does not know this guesthouse's prices, rules or bookings, so it
  would guess. Xiao Hai reads the real data, quotes the rule it used, calculates in code, and saves bookings.

## What It Can Do

| Function | Example question | How |
|---|---|---|
| Short facts (check-in, address, transport, parking, Wi-Fi) | "What time is check-in?" | tool `get_guesthouse_info` (database) |
| Rooms and services, prices, how many guests | "Do you have a room for 6 people?" | tool `search_rooms` (database) |
| Price of a stay (weekend +20%, 3+ nights 10% off, extras) | "How much are 3 nights in the Family Room with breakfast?" | tool `calculate_stay` (code) |
| Long house rules, with the source section | "If a typhoon comes on my check-in day, can I get my deposit back?" | tool `search_house_rules` (**RAG**) |
| Look up a booking (booking number + last 4 phone digits) | "Check my booking B1003, phone ends with 0917." | tool `get_booking` (database, read) |
| Book a room after the guest confirms | "Book the Twin Room from 2026-12-10 for 2 nights ..." | tool `make_booking` (database, **write**) |
| Cancel a booking, refund by the cancellation rule | "I want to cancel booking B1009, phone 1290." | tool `cancel_booking` (database, **write**) |
| Hand over to a person | "The air conditioner was broken, I want a refund." | tool `transfer_to_human` |

## How It Works

```text
guest message ─► agent (LLM with tool calling, at most 6 steps)
                   ├─ get_guesthouse_info / search_rooms ─► SQLite: shop_info, rooms
                   ├─ calculate_stay ───────────────────────► Python maths
                   ├─ search_house_rules ───────────────────► RAG: house_rules.md → 15 chunks → embeddings → top 2
                   ├─ get_booking / make_booking / cancel_booking ─► SQLite: bookings (read / write)
                   └─ transfer_to_human ────────────────────► front-desk phone and LINE
                 ◄─ answer + tool trace (right-hand panel)
```

- **Chat model:** `qwen2.5:14b` through Ollama's OpenAI-compatible API on a Google Colab T4 GPU
  (context 16,384 tokens, temperature 0). NKUST AI Hub or Gemini also work; see `.env.example`.
- **Embedding model (RAG):** `bge-m3` through Ollama (understands Chinese and English).
- **RAG:** `house_rules.md` is cut on its `## ` headings into 15 sections. Each section is embedded once when the
  program starts. A question is embedded, compared with every section by cosine similarity, and the best
  2 sections with a score of at least `MIN_SCORE = 0.50` are given to the model (chosen from `rag_scores.txt`: rule
  questions scored 0.53-0.74, off-topic questions 0.25-0.50). Below that score the agent says it does not
  have the information. The trace panel shows the section titles and scores, and the answer names its source.
- **Database (SQLite, built from CSV files at every start; `guesthouse.db` is not in the repository):**

  | Table | File | Rows |
  |---|---|---|
  | `shop_info` | `shop_info.csv` | 14 short facts |
  | `rooms` | `rooms.csv` | 8 rooms + 9 services |
  | `bookings` | `bookings.csv` | 10 fake bookings; new and cancelled bookings are written here |

- **Safety rules in code, not only in the prompt:**
  - a booking is shown only when the booking number *and* the last 4 phone digits match;
  - `make_booking` and `cancel_booking` first return a summary; the row is written only when the guest says yes in
    the **next** message, and the program saves exactly the summary the guest saw (the model cannot change the room);
  - past dates are refused; the refund (7+ days: all, 3-6 days: 50%, 2 days or less: 0) and the weekend nights are
    calculated in code; only the listed tools can run;
  - the agent keeps its earlier tool calls and results for the next turn, and the program adds a language reminder
    as the last message;
  - if the model answers a question that the house rules clearly cover (score ≥ 0.50) **without any tool**, the
    program runs the RAG search itself, shows it in the trace as "added by the program", and the model answers again.
- **System prompt:** five parts (role and goal, scope, interaction rules, refusals with exact sentences, opening line).
- **Chat screen:** Gradio, with a chat tab (greeting, sample questions in English and Chinese, tool trace) and a
  **Bookings** tab that shows the `bookings` table, so you can see that a booking is really saved.

## Sample Conversations

| | |
|---|---|
| RAG answer with its source | ![RAG](images/screenshot-rag.png) |
| Booking a room (write) | ![booking](images/screenshot-booking.png) |
| The Bookings tab after booking B1011 and cancelling B1009 | ![table](images/screenshot-table.png) |
| Not making things up | ![not found](images/screenshot-notfound.png) |

## Tests

```bash
python test_tool_choice.py   # 16 questions: does the agent choose the right tool (incl. RAG and booking tools)?
python test_scope.py         # 11 questions: off-topic, prompt attacks, role-play, missing booking, free night, past date
```

Final result (qwen2.5:14b + bge-m3 on Colab T4): **16/16** and **11/11**. The first versions scored 15/16 and 8/11;
the misses (a Chinese e-cigarette question with no tool, a printed fake "system prompt", a pool question sent to a
person without searching) were fixed with clearer routing and the checks in code above.

![Test results](images/test-results.png)

## Project Files

```text
cijin-seaview-guesthouse-rag/
├── app.py               # database, RAG, tools, agent loop and the Gradio page
├── house_rules.md       # the long house rules searched by RAG (15 sections, fake)
├── shop_info.csv        # short facts
├── rooms.csv            # rooms and services
├── bookings.csv         # fake bookings
├── test_tool_choice.py  # tool-choice test
├── test_scope.py        # scope and "do not make things up" test
├── rag_scores.txt       # RAG scores of 12 test questions (used to choose MIN_SCORE)
├── requirements.txt
├── .env.example         # sample settings (no real key)
├── .gitignore           # keeps .env and *.db out of git
└── images/              # screenshots
```

## How to Install and Run It

1. Download the project and install the packages (Python 3.10 or newer):

   ```bash
   git clone https://github.com/dobreira123/cijin-seaview-guesthouse-rag.git
   cd cijin-seaview-guesthouse-rag
   pip install -r requirements.txt
   ```

2. Make your settings file: copy `.env.example` to `.env` (`copy` on Windows, `cp` on macOS/Linux) and choose a chat
   model and an embedding model. With Ollama, first run `ollama pull qwen2.5:14b` and `ollama pull bge-m3`.
   With NKUST AI Hub or Gemini, put your own key after `API_KEY=` and `EMBEDDING_API_KEY=`.

3. Start it, then open the link in the console (usually <http://127.0.0.1:7860>):

   ```bash
   python app.py
   ```

   On Google Colab, run `GRADIO_SHARE=1 python app.py` to get a public link.

## Notes

- The guesthouse, its rules, prices, names, phone numbers and bookings are **all made up** for this project.
- `.env` holds keys and is **not** in this repository.
- Limits: there is no calendar of free rooms per date, and no payment. Next steps: an availability calendar,
  LINE integration, and a staff login for the Bookings tab.
