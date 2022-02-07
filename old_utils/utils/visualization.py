from spacy import displacy

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

def highlight_segments(id_example, data, texts=None):
    if texts is None:
        row = data[data["id"] == id_example].iloc[0]
        text = row["text"]
        ents = [
            {"start": pos[0], "end": pos[1], "label": label}
            for pos, label in zip(row["positions"], row["labels"])
            if label in ENTS
        ]
    else:
        text = texts[id_example]
        ents = []
        for _, row in data[data["id"] == id_example].iterrows():
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
