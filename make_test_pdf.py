import fitz  # PyMuPDF

def add_text(page, text, x, y, size=12):
    page.insert_text((x, y), text, fontsize=size)

def main(out_path="sample_task.pdf"):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter

    y = 60
    add_text(page, "Prompt", 50, y, 14); y += 18
    add_text(page, "User wants a weather app UI.", 60, y); y += 30

    # Side anchors (A/B) to support future per-side logic
    add_text(page, "Side A", 50, y, 13); y += 18
    add_text(page, "A: clean navbar, big temp card, hourly chips", 60, y); y += 24
    add_text(page, "Side B", 50, y, 13); y += 18
    add_text(page, "B: dense layout, many toggles, small font", 60, y); y += 30

    # Q1 (per-side yes/partial/no). We’ll show as lines the extractor can parse.
    add_text(page, "Q1:", 50, y, 13); y += 18
    add_text(page, "[x] Yes-fully    [ ] Yes-partially    [ ] No", 60, y); y += 28

    # Q2 (comparative) with options list and a Selected: line
    add_text(page, "Q2:", 50, y, 13); y += 18
    add_text(page, "Options: A is much better | A is better | about the same | B is better | B is much better | both are bad", 60, y); y += 18
    add_text(page, "Selected: B is better", 60, y); y += 24
    add_text(page, "Q2 explanation", 50, y, 13); y += 18
    add_text(page, "B uses clearer contrast and spacing; typography hierarchy is stronger.", 60, y); y += 28

    # Q3
    add_text(page, "Q3:", 50, y, 13); y += 18
    add_text(page, "Options: A is better | B is better | tie | both are bad", 60, y); y += 18
    add_text(page, "Selected: A is better", 60, y); y += 24
    add_text(page, "Q3 explanation", 50, y, 13); y += 18
    add_text(page, "A covers hourly and daily with fewer clicks; error empty states are present.", 60, y); y += 28

    # Q4
    add_text(page, "Q4:", 50, y, 13); y += 18
    add_text(page, "Options: A is better | B is better | tie | both are bad", 60, y); y += 18
    add_text(page, "Selected: A is better", 60, y); y += 24
    add_text(page, "Q4 explanation", 50, y, 13); y += 18
    add_text(page, "A feels faster and the input validation is clearer.", 60, y); y += 28

    # Q5 + explanation (also where our simple rating parser looks)
    add_text(page, "Q5:", 50, y, 13); y += 18
    add_text(page, "Options: A is better | B is better | tie | both are bad", 60, y); y += 18
    add_text(page, "Selected: A is better", 60, y); y += 24
    add_text(page, "Q5 explanation", 50, y, 13); y += 18
    add_text(page, "Overall A is better: clearer hierarchy, broader features, better functionality. Rating 4/5.", 60, y); y += 20

    doc.save(out_path)
    doc.close()
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
