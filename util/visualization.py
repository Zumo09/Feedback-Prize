import pandas as pd

COLORS = {
    "Position": "\033[31m",  # Red
    "Lead": "\033[32m",
    "Concluding Statement": "\033[33m",  # Yellow
    "Evidence": "\033[34m",  # Blue
    "Claim": "\033[35m",
    "Counterclaim": "\033[36m",
    "Rebuttal": "\033[95m",
    "bold": "\033[1m",
    "end": "\033[0m",
}


def print_segments(id_example: str, text: str, tags: pd.DataFrame):
    tags = tags[tags["id"] == id_example]
    try:
        labels = tags["class"]
    except KeyError:
        labels = tags["discourse_type"]

    indexes = [[int(i) for i in pred.split()] for pred in tags["predictionstring"]]

    boxes = [(i[0], i[-1]) for i in indexes]

    discourse = "end"
    for idx, word in enumerate(text.split()):
        color = "end"
        for (s, e), label in zip(boxes, labels):
            if s <= idx <= e:
                color = label
                break

        if discourse != color:
            if discourse != "end":
                print("[" + discourse + "]")
            else:
                print()

        discourse = color

        print(COLORS[color] + word, end=" ")