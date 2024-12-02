def plot_angle_3d(self, ax, origin, v1, v2, angle, num_points=1000, radius=0.5, **kwargs):
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
