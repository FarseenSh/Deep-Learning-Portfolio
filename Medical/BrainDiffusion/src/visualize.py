"""
3D visualization utilities for brain volumes.

This module provides Plotly-based visualization functions for 3D brain volumes,
diffusion processes, and comparison between real and generated samples.
"""

import numpy as np
import plotly.graph_objects as go
import torch
import torchio as tio
import nibabel as nib
from pathlib import Path


def visualize_brain_volume(volume, threshold=0.1, colorscale='Gray', title="3D Brain Volume"):
    """Create 3D visualization of brain volume.
    
    Args:
        volume: 3D numpy array or tensor of brain volume
        threshold: Minimum value to display
        colorscale: Plotly colorscale
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Ensure volume is numpy array with correct shape
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    
    # Remove batch and channel dimensions if present
    if volume.ndim == 5:  # [B, C, D, H, W]
        volume = volume[0, 0]
    elif volume.ndim == 4:  # [C, D, H, W]
        volume = volume[0]
        
    # Create meshgrid for 3D coordinates
    X, Y, Z = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
    
    # Create Plotly figure
    fig = go.Figure(data=go.Volume(
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=threshold,
        isomax=volume.max() * 0.8,  # Scale down max to see internal structures
        opacity=0.1,  # Adjust for visualization clarity
        surface_count=25,  # Adjust for smoother surface
        colorscale=colorscale
    ))
    
    # Configure layout
    fig.update_layout(
        title=title,
        width=800, 
        height=800,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    
    return fig


def visualize_diffusion_process(volume_timesteps, threshold=0.1):
    """Visualize brain volume across diffusion timesteps.
    
    Args:
        volume_timesteps: List of (timestep, volume) pairs
        threshold: Minimum value to display
        
    Returns:
        Plotly figure with slider for timesteps
    """
    # Create subplot grid based on number of timesteps
    fig = go.Figure()
    
    for i, (t, vol) in enumerate(volume_timesteps):
        # Prepare volume data
        if isinstance(vol, torch.Tensor):
            vol = vol.detach().cpu().numpy()
            
        # Remove batch and channel dimensions if present
        if vol.ndim == 5:  # [B, C, D, H, W]
            vol = vol[0, 0]
        elif vol.ndim == 4:  # [C, D, H, W]
            vol = vol[0]
            
        # Create meshgrid for 3D coordinates
        X, Y, Z = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
        
        # Add volume trace
        fig.add_trace(go.Volume(
            x=X.flatten(), 
            y=Y.flatten(), 
            z=Z.flatten(),
            value=vol.flatten(),
            isomin=threshold,
            isomax=vol.max() * 0.8,
            opacity=0.1,
            surface_count=15,
            colorscale='Gray',
            visible=(i==0),  # Only make first timestep visible initially
            name=f"t={t}"
        ))
    
    # Create slider for timesteps
    steps = []
    for i in range(len(volume_timesteps)):
        step = dict(
            method="update",
            args=[{"visible": [j == i for j in range(len(volume_timesteps))]}],
            label=f"t={volume_timesteps[i][0]}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        steps=steps,
        currentvalue={"prefix": "Timestep: "},
        pad={"t": 50}
    )]
    
    # Configure layout
    fig.update_layout(
        title="Diffusion Process Visualization",
        width=800, 
        height=800,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        sliders=sliders
    )
    
    return fig


def compare_real_generated(real_vol, gen_vol, threshold=0.1):
    """Compare real and generated brain volumes.
    
    Args:
        real_vol: Real brain volume
        gen_vol: Generated brain volume
        threshold: Minimum value to display
        
    Returns:
        Plotly figure
    """
    # Prepare volumes
    if isinstance(real_vol, torch.Tensor):
        real_vol = real_vol.detach().cpu().numpy()
    if isinstance(gen_vol, torch.Tensor):
        gen_vol = gen_vol.detach().cpu().numpy()
        
    # Remove batch and channel dimensions if present
    if real_vol.ndim == 5:  # [B, C, D, H, W]
        real_vol = real_vol[0, 0]
    elif real_vol.ndim == 4:  # [C, D, H, W]
        real_vol = real_vol[0]
        
    if gen_vol.ndim == 5:  # [B, C, D, H, W]
        gen_vol = gen_vol[0, 0]
    elif gen_vol.ndim == 4:  # [C, D, H, W]
        gen_vol = gen_vol[0]
    
    # Create figure with two subplots
    fig = go.Figure()
    
    # Real volume (blue)
    X, Y, Z = np.mgrid[0:real_vol.shape[0], 0:real_vol.shape[1], 0:real_vol.shape[2]]
    fig.add_trace(go.Volume(
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(),
        value=real_vol.flatten(),
        isomin=threshold,
        isomax=real_vol.max() * 0.8,
        opacity=0.1,
        surface_count=15,
        colorscale='Blues',
        name='Real',
        visible=True
    ))
    
    # Generated volume (red)
    X, Y, Z = np.mgrid[0:gen_vol.shape[0], 0:gen_vol.shape[1], 0:gen_vol.shape[2]]
    fig.add_trace(go.Volume(
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(),
        value=gen_vol.flatten(),
        isomin=threshold,
        isomax=gen_vol.max() * 0.8,
        opacity=0.1,
        surface_count=15,
        colorscale='Reds',
        name='Generated',
        visible=True
    ))
    
    # Create buttons to toggle visibility
    buttons = [
        dict(label="Both",
             method="update",
             args=[{"visible": [True, True]}]),
        dict(label="Real Only",
             method="update",
             args=[{"visible": [True, False]}]),
        dict(label="Generated Only",
             method="update",
             args=[{"visible": [False, True]}]),
    ]
    
    # Configure layout
    fig.update_layout(
        title="Real vs. Generated Brain Volume",
        width=900, 
        height=700,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=buttons,
            direction="right",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            y=1.1,
            xanchor="left",
            yanchor="top"
        )]
    )
    
    return fig


def visualize_brain_slices(volume, slice_indices=None):
    """Create 2D orthogonal slices visualization.
    
    Args:
        volume: 3D brain volume
        slice_indices: Optional indices for each axis, defaults to middle
        
    Returns:
        Plotly figure with orthogonal slices
    """
    # Prepare volume
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
        
    # Remove batch and channel dimensions if present
    if volume.ndim == 5:  # [B, C, D, H, W]
        volume = volume[0, 0]
    elif volume.ndim == 4:  # [C, D, H, W]
        volume = volume[0]
    
    # Default to middle slices if not specified
    if slice_indices is None:
        slice_indices = [
            volume.shape[0] // 2,  # Axial
            volume.shape[1] // 2,  # Coronal
            volume.shape[2] // 2   # Sagittal
        ]
    
    # Extract slices
    axial_slice = volume[slice_indices[0], :, :]
    coronal_slice = volume[:, slice_indices[1], :]
    sagittal_slice = volume[:, :, slice_indices[2]]
    
    # Create figure with 3 subplots
    fig = go.Figure()
    
    # Add heatmaps for each slice
    fig.add_trace(go.Heatmap(
        z=axial_slice.T,  # Transpose for correct orientation
        colorscale='Gray',
        showscale=False,
        name='Axial'
    ))
    
    fig.add_trace(go.Heatmap(
        z=coronal_slice.T,  # Transpose for correct orientation
        colorscale='Gray',
        showscale=False,
        name='Coronal',
        visible=False
    ))
    
    fig.add_trace(go.Heatmap(
        z=sagittal_slice.T,  # Transpose for correct orientation
        colorscale='Gray',
        showscale=False,
        name='Sagittal',
        visible=False
    ))
    
    # Create buttons to switch between views
    buttons = [
        dict(label="Axial",
             method="update",
             args=[{"visible": [True, False, False]},
                   {"title": f"Axial Slice (z={slice_indices[0]})"}]),
        dict(label="Coronal",
             method="update",
             args=[{"visible": [False, True, False]},
                   {"title": f"Coronal Slice (y={slice_indices[1]})"}]),
        dict(label="Sagittal",
             method="update",
             args=[{"visible": [False, False, True]},
                   {"title": f"Sagittal Slice (x={slice_indices[2]})"}]),
    ]
    
    # Configure layout
    fig.update_layout(
        title=f"Axial Slice (z={slice_indices[0]})",
        width=600, 
        height=600,
        updatemenus=[dict(
            type="buttons",
            buttons=buttons,
            direction="right",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            y=1.1,
            xanchor="left",
            yanchor="top"
        )]
    )
    
    # Update axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig


def save_nifti(volume, output_file):
    """Save volume as NIfTI file.
    
    Args:
        volume: 3D numpy array or tensor
        output_file: Path to save the NIfTI file
    """
    # Prepare volume
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
        
    # Remove batch and channel dimensions if present
    if volume.ndim == 5:  # [B, C, D, H, W]
        volume = volume[0, 0]
    elif volume.ndim == 4:  # [C, D, H, W]
        volume = volume[0]
    
    # Create NIfTI image with identity affine
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume, affine)
    
    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    nib.save(nifti_img, output_path)
    print(f"Saved NIfTI file to {output_path}")


if __name__ == "__main__":
    # Test visualization functions
    import argparse
    
    parser = argparse.ArgumentParser(description="Test visualization functions")
    parser.add_argument("--nifti_file", type=str, required=True, help="Path to NIfTI file")
    
    args = parser.parse_args()
    
    # Load volume
    subject = tio.Subject(mri=tio.ScalarImage(args.nifti_file))
    volume = subject.mri[tio.DATA]
    
    # Visualize brain volume
    fig = visualize_brain_volume(volume)
    fig.show()
    
    # Visualize orthogonal slices
    fig = visualize_brain_slices(volume)
    fig.show()
    
    print("Visualization test complete!")
