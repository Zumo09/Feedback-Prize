from spacy import displacy
from my_dataset import MyDataset

# Credits for this part of visualisation _> https://www.kaggle.com/thedrcat

ENTS = (
    "Lead",
    "Position",
    "Evidence",
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Rebuttal",
)

DISCOURSE_TYPES_COLORS = {
    "Lead": "#8000ff",
    "Position": "#2b7ff6",
    "Evidence": "#2adddd",
    "Claim": "#80ffb4",
    "Concluding Statement": "d4dd80",
    "Counterclaim": "#ff8042",
    "Rebuttal": "#ff0000",
}

def highlight_segments(id_example: str, dataset: MyDataset):
    text = dataset.documents[id_example]
    ents = []
    for _, row in dataset.tags[dataset.tags["id"] == id_example].iterrows(): # type: ignore
        ents.append(
            {
                "start": int(row["discourse_start"]),
                "end": int(row["discourse_end"]),
                "label": row["discourse_type"],
            }
        )

    doc2 = {"text": text, "ents": ents, "title": id_example}
    options = {"ents": ENTS, "colors": DISCOURSE_TYPES_COLORS}
    displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)
