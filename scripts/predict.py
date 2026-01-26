#### Get top -k predictions for a query image
import argparse
import torch
from torch.utils.data import DataLoader
from prim_package import (
    SingleImageDataset,
    load_single_df,
    get_split,
    SiameseNet,
)
from scripts.utils import extract_embeddings
from prim_package.data_processing.transforms import get_eval_transforms
from prim_package.training.losses import pairwise_distance
# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
def parse_args():
        p = argparse.ArgumentParser("Predict top-k similar products for query images")
        p.add_argument("--csv", required=True, help="CSV with image_path,label,split")
        p.add_argument("--root_dir", default="")
        p.add_argument("--model_path", required=True)
        p.add_argument("--embed_dim", type=int, default=256)
        p.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
        p.add_argument("--batch_size", type=int, default=32)
        p.add_argument("--im_size", type=int, default=256)
        p.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
        p.add_argument("--out", type=str, required=True, help="Output file for predictions")
        
        return p.parse_args()
# ---------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = load_single_df([args.csv])
    query_df = get_split(df, "query").query('label == "id_502"').head(1)  # Take only the first row for single query
    gallery_df = get_split(df, "gallery")

    transform = get_eval_transforms(im_size=args.im_size)
    query_dataset = SingleImageDataset(query_df, root_dir=args.root_dir, transform=transform)
    gallery_dataset = SingleImageDataset(gallery_df, root_dir=args.root_dir, transform=transform)

    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = SiameseNet(embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Extract embeddings
    query_embs, query_labels, query_paths = extract_embeddings(model, query_loader, device)
    gallery_embs, gallery_labels, gallery_paths = extract_embeddings(model, gallery_loader, device)

    # Compute distances and get top-k predictions
    results = []
    for i in range(len(query_embs)):
        q_emb = query_embs[i].unsqueeze(0)  # Shape: (1, D)
        if args.distance == "cosine":
            dists = pairwise_distance(q_emb, gallery_embs, mode="cosine").squeeze(0)
        else:
            dists = pairwise_distance(q_emb, gallery_embs, mode="euclidean").squeeze(0)

        top_k_indices = torch.topk(dists, k=args.top_k, largest=False).indices.numpy()
        top_k_paths = [gallery_paths[idx] for idx in top_k_indices]
        top_k_labels = [gallery_labels[idx] for idx in top_k_indices]

        results.append({
            "query_path": query_paths[i],
            "query_label": query_labels[i],
            "top_k_paths": top_k_paths,
            "top_k_labels": top_k_labels,
        })

    # Print results
    for res in results:
        print(f"Query: {res['query_path']} (Label: {res['query_label']})")
        for rank, (path, label) in enumerate(zip(res["top_k_paths"], res["top_k_labels"]), start=1):
            print(f"  Rank {rank}: {path} (Label: {label})")
            print()
    print(f"Predictions finished.")


if __name__ == "__main__":
    main()