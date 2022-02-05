import math
from sklearn.preprocessing import OrdinalEncoder
from spacy import displacy
from torch import Tensor
from dataset import FPDataset

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


def highlight_segments_old(id_example: str, dataset: FPDataset):
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


def highlight_segments(text: str, labels: Tensor, boxes: Tensor, title: str = 'Document'):
    splitted_text = text.split()

    len_words_inc = [0]
    cont = 0
    for word in splitted_text:
        idx = text[cont:].find(word)
        cont += max(0, idx)
        len_words_inc.append(cont + len(word))

    ls = len(splitted_text)
    ents = [
        {
            "start": len_words_inc[math.ceil(start * ls)],
            "end": len_words_inc[math.ceil(end * ls) + 1],
            "label": label,
        }
    # TODO Fix 
        for label, (start, end) in zip(labels, boxes)
    ]

    doc2 = {"text": text, "ents": ents, "title": title}
    displacy.render(doc2, style="ent", options=OPTIONS, manual=True, jupyter=True)
