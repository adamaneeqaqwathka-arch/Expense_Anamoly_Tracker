# Design decisions (Step 0)

Fill in your final choice and the reason. These four decide the engine's interface.

| # | Question | Recommended | Your decision | Why |
|---|----------|-------------|---------------|-----|
| 1 | Fit on each uploaded file, or use one pretrained model? | Fit on the upload (minimum row count), pretrained only as a demo mode | | |
| 2 | How do we handle different CSV layouts? | Column mapping: required amount and date; optional category, merchant, card ID; auto-detect, user corrects | | |
| 3 | Do we store anything? | No database at first; SQLite later for run history and reviewer decisions | | |
| 4 | Where does it run? | One container for API and website; Streamlit as a second small service | | |
