import potpourri3d as pp3d
import trimesh
import time
import json
import numpy as np
import argparse

def shuffle_label(labels):
    num_label = len(np.unique(labels))
    shuffle_dict = {i: i for i in range(num_label)}

    # Get dictionary values and shuffle them randomly
    values = list(shuffle_dict.values())
    np.random.shuffle(values)
    # Update dictionary values
    for i, key in enumerate(shuffle_dict.keys()):
        shuffle_dict[key] = values[i]
    with open("shuffle_dict.json", 'w') as f:
        json.dump(shuffle_dict, f)
    labels = np.asarray([shuffle_dict[l] for l in labels])

    return labels


def place_spheres_on_vertices(mesh, vertex_indices, sphere_radius=0.01):
    spheres = []
    for index in vertex_indices:
        # Get the vertex position
        vertex_position = mesh.vertices[index]
        
        # Create a sphere
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
        
        # Move the sphere to the vertex position
        sphere.apply_translation(vertex_position)
        
        # Set the sphere color to red
        sphere.visual.vertex_colors = [255, 0, 0, 255]  # RGBA for red
        
        # Add the sphere to the list
        spheres.append(sphere)
    
    # Combine all spheres into a single mesh
    combined_spheres = trimesh.util.concatenate(spheres)
    
    # Combine the original mesh with the spheres
    final_mesh = trimesh.util.concatenate([mesh, combined_spheres])
    
    return final_mesh

def subdivide(mesh):
    # Note: This function concatenates the newly generated vertices to the end of the original vertices,
    # so the indices of the original vertices remain unaffected. However, please manually verify this
    # behavior to ensure that the version of trimesh you are using indeed performs this operation.
    v_, f_ = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    new_mesh = trimesh.Trimesh(v_, f_)
    return new_mesh

def test(mesh_path):
    with open('superset_smpl.json', 'r') as f:
        superset_smpl = json.load(f)

    t1 = time.time()

    markers_indices = list(superset_smpl.values())
    mesh = trimesh.load_mesh(mesh_path, maintain_order=True, process=False)
    new_mesh = subdivide(mesh)

    solver = pp3d.MeshHeatMethodDistanceSolver(new_mesh.vertices, new_mesh.faces)
    m_i_ = np.zeros((len(markers_indices), len(new_mesh.vertices)))
    print(m_i_.shape)
    for m, marker_index in enumerate(markers_indices):
        dist = solver.compute_distance(marker_index)
        m_i_[m] = dist
    
    geodesic_distances = np.min(m_i_, axis=0)
    labels = np.argmin(m_i_, axis=0)

    shuffle_label_ = True
    if shuffle_label_:
        labels = shuffle_label(labels)


    print(labels.max(), labels.min(), len(np.unique(labels)))
    t2 = time.time()
    print("Time used:", t2 - t1)

    # visulization
    colors_gedesic_distances = trimesh.visual.color.interpolate(geodesic_distances, color_map='viridis')
    new_mesh.visual.vertex_colors = colors_gedesic_distances
    final_mesh_gedesic_distances = place_spheres_on_vertices(new_mesh, markers_indices)
    final_mesh_gedesic_distances.export(f"gedesic_distances_{mesh_path.split('/')[-1]}.obj")

    
    colors_labels = trimesh.visual.color.interpolate(labels, color_map='viridis')
    new_mesh.visual.vertex_colors = colors_labels
    final_mesh_label = place_spheres_on_vertices(new_mesh, markers_indices)
    final_mesh_label.export(f"labels_{mesh_path.split('/')[-1]}.obj")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str, default="smpl_mesh_original.obj", help="mesh path")
    args = parser.parse_args()
    mesh_path = args.mesh_path
    np.random.seed(42)
    test(mesh_path)