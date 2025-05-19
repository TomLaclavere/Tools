import numpy as np
import plotly.graph_objects as go
from getdist import plots, MCSamples

def plot_angle_3d(ax, origin, v1, v2, angle, num_points=1000, radius=0.5, **kwargs):
    """Plot angle 3d.
    
    General function to plot a 3D angle between two vectors v1 and v2.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Matplotlib 3D axis
    origin : array_like
        Position of the origin of the angle.
    v1 : array_like
        Vector 1.
    v2 : array_like
        Vector 2.
    angle : float
        Angle between v1 and v2, in radians.
    num_points : int, optional
        Number of points used to plot the angle, by default 100
    radius : float, optional
        Radiius from the origin at which the angle is plotted, by default 0.5
        
    Other Parameters
    ----------------
    kwargs : optional
        Any kwarg for plt.plot()
    """       
    
    #! This function can be moved in a more genral repostitory
    
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Create orthonormal basis for the plane containing v1 and v2
    normal = np.cross(v1_norm, v2_norm)
    if np.allclose(normal, 0):
        # Vectors are parallel, choose an arbitrary perpendicular vector
        normal = np.array([1, 0, 0]) if np.allclose(v1_norm, [0, 1, 0]) else np.cross(v1_norm, [0, 1, 0])
    normal = normal / np.linalg.norm(normal)
    
    angles = np.linspace(0, angle, num_points)
    arc_points = np.zeros((num_points, 3))
    
    for i, theta in enumerate(angles):
        rotated = v1_norm * np.cos(theta) + \
                np.cross(normal, v1_norm) * np.sin(theta) + \
                normal * np.dot(normal, v1_norm) * (1 - np.cos(theta))
        arc_points[i] = origin + radius * rotated
        
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], **kwargs)

def plot_vector(fig, pos, vector, color='blue', name='vector', show_arrow=True, arrow_size = 0.2):
    """Plot vector with plotly.
    
    General method to plot a vector with arrow using plotly.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    pos : array_like
        Position of the vector.
    vector : array_like
        Vector to plot.
    color : str, optional
        Color of the vector, by default 'blue'
    name : str, optional
        Name of the vector, by default 'vector'
    show_arrow : bool, optional
        Show or not the arrow, by default True
    arrow_size : float, optional
        Vector's arrow size, by default 0.2
    """
    
    ### Coordiantes of the two points defining the vector
    start = pos
    end = pos + vector
    
    ### Build unitary vector
    vector_unit = vector / np.linalg.norm(vector)

    ### Plot the segment between the two points
    fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], 
            y=[start[1], end[1]], 
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=color, width=2),
            name=name,
            text=[name, name],
            hovertemplate=(
                    "<b>%{text}</b><br>"
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<br>"
                    "Z: %{z:.2f}<extra></extra>" 
                    )
        ))

    ### Plot the arrowhead
    if show_arrow:
        if vector_unit[0] != 0 or vector_unit[1] != 0:
            # General case: construct perpendicular vectors
            ortho1 = np.cross(vector_unit, [0, 0, 1])
        else:
            # Special case: vector is along z-axis
            ortho1 = np.cross(vector_unit, [1, 0, 0])
        
        ortho1 /= np.linalg.norm(ortho1)  
        # Compute the second orthogonal vector
        ortho2 = np.cross(vector_unit, ortho1)  
        ortho2 /= np.linalg.norm(ortho2)  
        # Base of the arrowhead
        tip_base = np.array(end) - arrow_size * vector_unit

        # Compute the points for the arrowhead
        point1 = tip_base + arrow_size * 0.5 * ortho1
        point2 = tip_base - arrow_size * 0.5 * ortho1

        # Add the arrowhead segments
        for point in [point1, point2]:
            fig.add_trace(go.Scatter3d(
                x=[end[0], point[0]],
                y=[end[1], point[1]],
                z=[end[2], point[2]],
                mode='lines',
                line=dict(color=color, width=5),
                showlegend=False,
                hovertemplate=(
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<br>"
                    "Z: %{z:.2f}<extra></extra>" 
                    )
            ))

def triangle_plot(samples_flat, param_names):
    """Triangle plot.

    Functoin to plot the triangle plot of the samples. Add vertical and horizontal lines at the median of the samples.

    Parameters
    ----------
    samples_flat : array_like
        Flat array from emcee. The shape of the array is (n_walkers x n_samples, n_params).
    param_names : array_like
        Array of the parameter names.
    """
    mcsamples = MCSamples(samples=samples_flat, names=param_names, labels=param_names)

    best_fit = np.median(samples_flat, axis=0)

    g = plots.get_subplot_plotter()
    g.settings.num_plot_contours = 3
    g.triangle_plot(mcsamples, filled=True, markers=param_names)

    for i in range(g.subplots.shape[0]):
        for j in range(g.subplots.shape[1]):
            if i >= j:
                ax = g.subplots[i, j]
                ax.axvline(best_fit[j], color='grey', ls='--', lw=1.5)
                if i > j:
                    ax.axhline(best_fit[i], color='grey', ls='--', lw=1.5)
