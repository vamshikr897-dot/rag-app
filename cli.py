from rag_engine import ask, get_collection

if __name__ == "__main__":
    print("Loading collection...")
    collection = get_collection()
    print("Ready. Ask a question (or type 'quit').\n")

    while True:
        query = input("> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        result = ask(query, collection)
        print(f"\n{result['answer']}\n")
        print("Sources:")
        for s in result["sources"]:
            chapter = f"Chapter {s['chapter_number']}" if s["chapter_number"] else s["chapter_title"]
            print(f"  - {chapter} ({s['chapter_title']}), p.{s['page']}")
        print(f"[tokens: {result['input_tokens']} in / {result['output_tokens']} out]\n")
