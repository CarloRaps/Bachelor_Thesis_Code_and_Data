

#Imports
# `score` is the main BERTScore function
from bert_score import score

# torch is needed to (a) detect GPU availability and (b) work with the tensors of bertscore.
import torch

import pandas as pd
import os


# You can put as many pairs as you like.  Processing is batched automatically.
candidates = [
    """[insert LLM text]""", """[insert LLM text]"""
]
references = [
    """[insert CJEU Summary]""", """[insert CJEU Summary]""" 
]


MODEL = "microsoft/deberta-base-mnli"


# Automatically use a GPU (CUDA) if one is available otherwise cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model selection based on avaible computing power 

def get_best_model(device):
    models = [
        "microsoft/deberta-xlarge-mnli",
        "microsoft/deberta-large-mnli",
        "microsoft/deberta-base-mnli",
    ]
    for model in models:
        try:
            print(f"Trying {model}...")
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model)
            mdl = AutoModel.from_pretrained(model).to(device)
            del mdl
            torch.cuda.empty_cache()
            print(f"Using {model}")
            return model
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print(f"Not enough VRAM for {model}, trying next...")
            torch.cuda.empty_cache()
    raise RuntimeError("No model fit in VRAM")

MODEL = get_best_model(DEVICE)

# Summary of background info for testing

print("BERTScore Evaluation")

print(f"Model   : {MODEL}")
print(f"Device  : {DEVICE}")
print(f"# Pairs : {len(candidates)}")


# cands sets out the list of strings that represent the candidate (LLM's summary)
# refs sets out the list of strings that represent the refernce (CJEU summary)
# model_type sets model to previously set transformer
# lang set to en for english 
# verbose trouble shooting progress bar
# device sets it to the device selected before (GPU vs CPU)
# nthreads set to 4 to run well without causing stress
# batch_size setting number of pairs to 8
# rescale_with_baseline sets default band to something more readable (0-1)

P, R, F1 = score(
    cands=candidates,
    refs=references,
    model_type=MODEL,
    lang="en",
    verbose=True,
    device=DEVICE,
    nthreads=4,
    batch_size=8,
    rescale_with_baseline=True,
)


# results between each pair (snippet to east token chunks)

print("Per-Pair Results  (rescaled with baseline)")


for i, (cand, ref, p, r, f) in enumerate(
    zip(candidates, references, P.tolist(), R.tolist(), F1.tolist()), start=1
):
    print(f"\nPair {i}")
    print(f"  Candidate : {cand}")
    print(f"  Reference : {ref}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f:.4f}")



print("Mean Scores")

print(f"  Mean Precision : {P.mean().item():.4f}")
print(f"  Mean Recall    : {R.mean().item():.4f}")
print(f"  Mean F1        : {F1.mean().item():.4f}")

# adds outcome to a existing file each run in a new col. 
RESULTS_FILE = ""
#Insert filename above ().xslx)

mean_p  = P.mean().item()
mean_r  = R.mean().item()
mean_f1 = F1.mean().item()

if os.path.exists(RESULTS_FILE):
    df = pd.read_excel(RESULTS_FILE, index_col=0)
    trial_num = df.shape[1] + 1
else:
    df = pd.DataFrame(index=["Precision", "Recall", "F1"])
    trial_num = 1

df[f"Trial {trial_num}"] = [round(mean_p, 4), round(mean_r, 4), round(mean_f1, 4)]
df.to_excel(RESULTS_FILE)
print(f"\nTrial {trial_num} saved to {RESULTS_FILE}")
