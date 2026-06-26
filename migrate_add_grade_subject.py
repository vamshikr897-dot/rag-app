import config
from rag_engine import get_collection

BATCH_SIZE = 100


def migrate(grade: int = 8, subject: str = "science") -> int:
    collection = get_collection()
    existing = collection.get(include=["metadatas"])

    ids_to_update = []
    metadatas_to_update = []

    for record_id, metadata in zip(existing["ids"], existing["metadatas"]):
        if "grade" in metadata and "subject" in metadata:
            continue
        metadata = {**metadata, "grade": grade, "subject": subject}
        ids_to_update.append(record_id)
        metadatas_to_update.append(metadata)

    for start in range(0, len(ids_to_update), BATCH_SIZE):
        batch_ids = ids_to_update[start : start + BATCH_SIZE]
        batch_metadatas = metadatas_to_update[start : start + BATCH_SIZE]
        collection.update(ids=batch_ids, metadatas=batch_metadatas)

    return len(ids_to_update)


if __name__ == "__main__":
    updated_count = migrate(grade=8, subject=config.SUBJECT)
    print(f"Updated {updated_count} chunk(s) with grade/subject metadata.")
