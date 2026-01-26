#### Compute similarity scores between query and gallery images
import argparse
import torch
import csv
import time
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
        p = argparse.ArgumentParser("Compute similarity scores between query and gallery images")
        p.add_argument("--csv", required=True, help="CSV with image_path,label,split")
        p.add_argument("--root_dir", default="")
        p.add_argument("--model_path", required=True)
        p.add_argument("--embed_dim", type=int, default=256)
        p.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
        p.add_argument("--batch_size", type=int, default=32)
        p.add_argument("--im_size", type=int, default=256)
        p.add_argument("--out", type=str, required=True, help="Output CSV file for scores")
        p.add_argument("--save_details", action="store_true", help="Save individual scores for each gallery image")
        
        return p.parse_args()
# ---------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = load_single_df([args.csv])
    
    # Get all unique labels
    unique_labels = sorted(df['label'].unique())
    print(f"Found {len(unique_labels)} unique IDs: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")
    
    # Load model
    model = SiameseNet(embed_dim=args.embed_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    transform = get_eval_transforms(im_size=args.im_size)
    
    # Start timing prediction
    prediction_start_time = time.time()
    
    results = []
    
    # Loop through each unique ID
    for label_id in unique_labels:
        print(f"\nProcessing {label_id}...")
        
        # Get query and gallery for this ID
        query_df = get_split(df, "query").query(f'label == "{label_id}"')
        gallery_df = get_split(df, "gallery").query(f'label == "{label_id}"')
        
        if len(query_df) == 0:
            print(f"  No query images for {label_id}, skipping")
            continue
        if len(gallery_df) == 0:
            print(f"  No gallery images for {label_id}, skipping")
            continue
        
        print(f"  Query images: {len(query_df)}, Gallery images: {len(gallery_df)}")
        
        # Create datasets and loaders
        query_dataset = SingleImageDataset(query_df, root_dir=args.root_dir, transform=transform)
        gallery_dataset = SingleImageDataset(gallery_df, root_dir=args.root_dir, transform=transform)
        
        query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
        gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Extract embeddings
        query_embs, query_labels, query_paths = extract_embeddings(model, query_loader, device)
        gallery_embs, gallery_labels, gallery_paths = extract_embeddings(model, gallery_loader, device)
        
        # Compute distances and get average score for each query
        for i in range(len(query_embs)):
            q_emb = query_embs[i].unsqueeze(0)  # Shape: (1, D)
            if args.distance == "cosine":
                scores = pairwise_distance(q_emb, gallery_embs, mode="cosine").squeeze(0)
            else:
                scores = pairwise_distance(q_emb, gallery_embs, mode="euclidean").squeeze(0)
            
            # Compute average score
            avg_score = scores.mean().item()
            min_score = scores.min().item()
            max_score = scores.max().item()
            std_score = scores.std().item()
            
            results.append({
                "query_path": query_paths[i],
                "query_label": query_labels[i],
                "avg_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "std_score": std_score,
                "num_gallery": len(gallery_embs),
                "all_scores": scores.cpu().numpy() if args.save_details else None,
                "gallery_paths": gallery_paths if args.save_details else None,
                "gallery_labels": gallery_labels if args.save_details else None,
            })
    
    # End timing prediction
    prediction_time = time.time() - prediction_start_time
    
    print(f"\n{'='*60}")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Total queries processed: {len(results)}")
    print(f"{'='*60}")

    # Save results to CSV
    with open(args.out, 'w', newline='') as f:
        if args.save_details:
            writer = csv.writer(f)
            writer.writerow(["query_path", "query_label", "avg_score", "min_score", "max_score", "std_score", "num_gallery", "gallery_path", "gallery_label", "score", "prediction_time_seconds"])
            for res in results:
                for gal_path, gal_label, score in zip(res["gallery_paths"], res["gallery_labels"], res["all_scores"]):
                    writer.writerow([res["query_path"], res["query_label"], res["avg_score"], res["min_score"], res["max_score"], res["std_score"], res["num_gallery"], gal_path, gal_label, score, prediction_time])
        else:
            writer = csv.DictWriter(f, fieldnames=["query_path", "query_label", "avg_score", "min_score", "max_score", "std_score", "num_gallery", "prediction_time_seconds"])
            writer.writeheader()
            for res in results:
                writer.writerow({
                    "query_path": res["query_path"],
                    "query_label": res["query_label"],
                    "avg_score": res["avg_score"],
                    "min_score": res["min_score"],
                    "max_score": res["max_score"],
                    "std_score": res["std_score"],
                    "num_gallery": res["num_gallery"],
                    "prediction_time_seconds": prediction_time,
                })

    # Print results summary
    print(f"\nSample Results:")
    for res in results[:5]:  # Show first 5
        print(f"  {res['query_label']}: avg={res['avg_score']:.4f}, min={res['min_score']:.4f}, max={res['max_score']:.4f}")
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more")
    print(f"\nScores saved to: {args.out}")


if __name__ == "__main__":
    main()