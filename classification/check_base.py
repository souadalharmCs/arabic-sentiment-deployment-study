import statistics

from transformers import AutoModel, AutoModelForSequenceClassification

CANDIDATES = ["UBC-NLP/MARBERT", "UBC-NLP/MARBERTv2"]


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def compare(sd_a, sd_b):
    return [rel(v, sd_b[k]) for k, v in sd_a.items()
            if k in sd_b and sd_b[k].shape == v.shape and v.dim() > 1]


ft = AutoModelForSequenceClassification.from_pretrained("./model/checkpoint-680").bert
sd_ft = ft.state_dict()
print("fine-tuned checkpoint: vocab", ft.config.vocab_size, "| layers", ft.config.num_hidden_layers)

bases = {}
for name in CANDIDATES:
    try:
        bases[name] = AutoModel.from_pretrained(name).state_dict()
    except Exception as e:
        print(f"\n{name}: NOT LOADED ({type(e).__name__}) - not in the local cache")
        continue
    d = compare(sd_ft, bases[name])
    if not d:
        print(f"\n{name}: no tensors with matching names/shapes (different vocabulary?)")
        continue
    emb = "embeddings.word_embeddings.weight"
    ok = emb in bases[name] and bases[name][emb].shape == sd_ft[emb].shape
    emb_d = rel(sd_ft[emb], bases[name][emb]) if ok else float("nan")
    print(f"\n{name}: {len(d)} weight matrices compared")
    print(f"  median relative difference: {statistics.median(d):.4f}")
    print(f"  max relative difference:    {max(d):.4f}")
    print(f"  word embeddings:            {emb_d:.4f}")

if len(bases) == 2:
    a, b = list(bases)
    d = compare(bases[a], bases[b])
    if d:
        print(f"\nfor scale, distance between {a} and {b}: median {statistics.median(d):.4f}")
