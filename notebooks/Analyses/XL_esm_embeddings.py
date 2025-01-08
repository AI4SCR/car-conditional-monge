from pathlib import Path

import pandas as pd
import scanpy as sc
import torch
from transformers import AutoTokenizer, EsmModel

adata = sc.read_h5ad("/path/to/data/20240229_from_rocio_for_manuscript.h5ad")
all_cars = adata.obs["CAR_Variant"].unique().tolist()

data_dir = Path("/path/to/sequences/")
domains = [
    "CD28",
    "41BB",
    "CTLA4",
    "IL15RA",
    "CD40",
    "z",
    "domain_to_domain",
    "domain_to_z",
    "extracell_and_TMD_CARs",
    "NANANA_tiny_tail",
]
linker_aa = "SA"
all_domains = {}
for d in domains:
    with open(data_dir / f"{d}.txt") as f:
        lines = f.readlines()
        lines = "".join(lines)
        all_domains[d] = lines
        print(d, len(lines))
all_domains["NA"] = ""

full_cars = {}
tails = {}
for car in all_cars:
    domain1, domain2, domain3 = car.split("-")
    if domain3 == "NA":
        tail = all_domains["NANANA_tiny_tail"]
    else:
        tail = all_domains[domain1] + all_domains[domain2] + all_domains[domain3]
    full_car = all_domains["extracell_and_TMD_CARs"] + tail
    full_cars[car] = full_car
    tails[car] = tail

seqs = pd.DataFrame(pd.Series(full_cars).rename("full_car")).merge(
    pd.Series(tails).rename("tail"), left_index=True, right_index=True
)

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")

inputs = tokenizer(seqs["tail"].tolist(), return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
avg_embed_seq_tail = last_hidden_states.numpy().mean(axis=2)
avg_embed_dim_tail = last_hidden_states.numpy().mean(axis=1)

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t48_15B_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t48_15B_UR50D")

inputs = tokenizer(seqs["full_car"].tolist(), return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
avg_embed_seq = last_hidden_states.numpy().mean(axis=2)
avg_embed_dim = last_hidden_states.numpy().mean(axis=1)


full_car_avg_seq = pd.DataFrame(avg_embed_seq, index=seqs.index).T
full_car_avg_dim = pd.DataFrame(avg_embed_dim, index=seqs.index).T

tail_avg_seq = pd.DataFrame(avg_embed_seq_tail, index=seqs.index).T
tail_avg_dim = pd.DataFrame(avg_embed_dim_tail, index=seqs.index).T

print(full_car_avg_seq.shape)
print(full_car_avg_dim.shape)
print(tail_avg_seq.shape)
print(tail_avg_dim.shape)

full_car_avg_seq.to_csv("/path/to/embedding/esm2_t48_15B_UR50D_full_seq")
full_car_avg_dim.to_csv("/path/to/embedding/esm2_t48_15B_UR50D_full_dim")
tail_avg_seq.to_csv("/path/to/embedding/esm2_t48_15B_UR50D_tail_seq")
tail_avg_dim.to_csv("/path/to/embedding/esm2_t48_15B_UR50D_tail_dim")
