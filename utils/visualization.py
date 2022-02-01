from spacy import displacy
from my_dataset import MyDataset

# Credits for this part of visualisation _> https://www.kaggle.com/thedrcat

OPTIONS = {
    "ents": (
        "Lead",
        "Position",
        "Evidence",
        "Claim",
        "Concluding Statement",
        "Counterclaim",
        "Rebuttal",
    ),
    "colors": {
        "Lead": "#8000ff",
        "Position": "#2b7ff6",
        "Evidence": "#2adddd",
        "Claim": "#80ffb4",
        "Concluding Statement": "d4dd80",
        "Counterclaim": "#ff8042",
        "Rebuttal": "#ff0000",
    },
}


def highlight_segments_old(id_example: str, dataset: MyDataset):
    text = dataset.documents[id_example]
    ents = []
    for _, row in dataset.tags[dataset.tags["id"] == id_example].iterrows():  # type: ignore
        ents.append(
            {
                "start": int(row["discourse_start"]),
                "end": int(row["discourse_end"]),
                "label": row["discourse_type"],
            }
        )

    doc2 = {"text": text, "ents": ents, "title": id_example}

    displacy.render(doc2, style="ent", options=OPTIONS, manual=True, jupyter=True)


def highlight_segments(index: int, dataset: MyDataset):
    text, data = dataset[index]  # type: ignore
    ents = []
    splitted_text = text.split()
    for label, center, length in data:
        real_c = sum(len(s) + 1 for s in splitted_text[:int(center)])
        real_l = sum(len(s) + 1 for s in splitted_text[int(center - length / 2): int(center + length / 2)])
        ents.append(
            {
                "start": int(real_c - real_l / 2),
                "end": int(real_c + real_l / 2),
                "label": dataset.encoder.categories_[0][int(label)],
            }
        )

    doc2 = {"text": text, "ents": ents, "title": dataset.documents.index[index]}
    displacy.render(doc2, style="ent", options=OPTIONS, manual=True, jupyter=True)
