import argparse
import os
import numpy as np
import torch
from models.models_pointcloud import GT_network_equiv
from data_utils.GT_dataloader_center import GTDataset
from torch.utils.data import DataLoader
import trimesh
import json
from models.fit_SMPL import fit_smpl

def load_model(args):
    """Load the trained model"""
    model = GT_network_equiv(option=args).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    return model

def preprocess_scan(scan_path):
    """Load and center the scan mesh"""
    scan_mesh = trimesh.load_mesh(scan_path, process=False, maintain_order=True)
    scan_vertices = scan_mesh.vertices
    
    # Center the scan
    scan_min_xyz = np.min(scan_vertices, axis=0)
    scan_max_xyz = np.max(scan_vertices, axis=0)
    scan_center = (scan_min_xyz + scan_max_xyz) / 2.0
    centered_vertices = scan_vertices - scan_center
    
    # Create new centered mesh
    centered_mesh = scan_mesh.copy()
    centered_mesh.vertices = centered_vertices
    
    return centered_mesh, scan_center

def sample_points_from_mesh(mesh, num_points=5000):
    """Sample points from mesh surface"""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def predict_smpl(args, model, points, gender='neutral'):
    """Predict SMPL parameters from input points"""
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(args.device)
        
        # Model prediction
        PRED_ITEMS = ["confidence", "direction", "magnitude"]
        results, selected_indexs = model(points_tensor, pred_items=PRED_ITEMS, direction_mode="standard_vector")
        
        # Process results
        pred_part_labels = results["part_labels"]
        _, pred_part_labels = torch.max(pred_part_labels, -1)
        pred_confidences = results["confidences"]
        pred_directions = results["direction"]
        pred_magnitudes = results["magnitude"]
        
        pred_vectors = pred_directions * pred_magnitudes / args.scale_magnitude
        pred_inner_points = points_tensor - pred_vectors
        
        # Fit SMPL
        final_mesh_list, _, _, output_smpl_info = fit_smpl(
            args, pred_inner_points, pred_part_labels, pred_confidences, gender
        )
        
        return final_mesh_list[0], output_smpl_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_path", type=str, required=True, help="Path to input scan OBJ file")
    parser.add_argument("--gender", type=str, default="neutral", choices=["neutral", "male", "female"], 
                       help="Gender of the subject")
    parser.add_argument("--model_path", type=str, default="all_experiments/experiments/EPN_layer_2_radius_0.4_aug_so3_num_point_5000/mix_1_full/checkpoints/model_epochs_00000032.pth", help="Path to trained model")
    parser.add_argument("--markerset_path", default="datafolder/useful_data_cape/superset_smpl.json", 
                       type=str, help="Path to markerset JSON file")
    parser.add_argument("--output_folder", type=str, default="output", help="Output directory")
    parser.add_argument("--num_point", type=int, default=5000, help="Number of points to sample from scan")
    parser.add_argument("--scale_magnitude", type=int, default=10, help="Scale for magnitude")
    parser.add_argument("--EPN_input_radius", type=float, default=0.4)
    parser.add_argument("--EPN_layer_num", type=int, default=2)
    
    args = parser.parse_args()
    
    # Setup device
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    # Load markerset
    with open(args.markerset_path, 'r') as f:
        args.markerset = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Load model
    model = load_model(args)
    
    # Load and preprocess scan
    centered_mesh, original_center = preprocess_scan(args.scan_path)
    
    # Sample points from scan
    points = sample_points_from_mesh(centered_mesh, args.num_point)
    
    # Predict SMPL
    pred_smpl_mesh, smpl_info = predict_smpl(args, model, points, args.gender)
    
    # Apply inverse translation to move back to original space
    pred_smpl_vertices = pred_smpl_mesh.vertices + original_center
    final_smpl_mesh = pred_smpl_mesh.copy()
    final_smpl_mesh.vertices = pred_smpl_vertices
    
    # Save results
    scan_name = os.path.splitext(os.path.basename(args.scan_path))[0]
    output_path = os.path.join(args.output_folder, f"{scan_name}_pred_smpl.obj")
    final_smpl_mesh.export(output_path)
    
    print(f"Predicted SMPL mesh saved to: {output_path}")

if __name__ == "__main__":
    main()
