import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

def ini_final_clusters(Fmap, n_clusters, labels, img_name, param, e):
    positions_ini=Fmap[0,:,:]
    positions_end=Fmap[-1,:,:]


    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Define a color map
    colors = plt.get_cmap("viridis", n_clusters)
    cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]  # Names for legend

    # First subplot
    scatter1 = axes[0].scatter(positions_ini[0,:], positions_ini[1,:], c=labels, cmap=colors, vmin=0, vmax=n_clusters-1)
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].set_title("Initial distribution of the clusters")

    # Second subplot
    scatter2 = axes[1].scatter(positions_end[0,:], positions_end[1,:], c=labels, cmap=colors, vmin=0, vmax=n_clusters-1)
    axes[1].set_xlabel("X Position")
    axes[1].set_title("Final distribution of the clusters")

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cluster_names[j],
                            markerfacecolor=colors(j), markersize=8) for j in range(n_clusters)]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

    # Main title
    plt.suptitle(f"{param}_{n_clusters}clusters_{e}spars", fontsize=16)
    plt.subplots_adjust(right=0.8) 
    # Save the figure
    plt.savefig(f"{img_name}")
    plt.show()
    
def gif_clusters(Fmap, n_clusters, labels, img_name, param, e):
    fig, ax = plt.subplots(figsize=(8,6))

    # Define a color map
    colors = plt.get_cmap("viridis", n_clusters)
    cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]  # Names for legend

    ymax = Fmap[:,1,:].max()
    ymin = Fmap[:,1,:].min()
    xmax = Fmap[:,0,:].max()
    xmin = Fmap[:,0,:].min()

    def animate(i):
        ax.clear()
        # Scatter plot with colors based on labels
        scatter = plt.scatter(Fmap[i,0,:], Fmap[i,1,:], c=labels, cmap=colors)

        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=cluster_names[j],
                            markerfacecolor=colors(j), markersize=8) for j in range(n_clusters)]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1))

        # Set fixed axis limits
        ax.set_xlim(xmin,xmax) 
        ax.set_ylim(ymin,ymax)  

        #Add labels
        ax.set_title(f"Cluster at timestep {i+1}")
        ax.set_xlabel("Degrees W")
        ax.set_ylabel("Degrees E")

    # Filter frames to only include multiples of 10
    frames_to_plot = [i for i in range(Fmap[:,:,:].shape[0]) if i % 5 == 0]

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=frames_to_plot, repeat=False)
   
    plt.subplots_adjust(right=0.8) 
    # Save the animation as a video file
    ani.save(img_name,writer='pillow',fps=10)


def ini_final_clusters_landmask(Fmap, n_clusters, labels, img_name, e, x, y, mask_interpol):
    positions_ini=Fmap[0,:,:]
    positions_end=Fmap[-1,:,:]
    
    ymax = Fmap[:,1,:].max()
    ymin = Fmap[:,1,:].min()
    xmax = Fmap[:,0,:].max()
    xmin = Fmap[:,0,:].min()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Define a color map for the clusters
    colors = plt.get_cmap("tab20", n_clusters)
    cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]  # Names for legend

    # Define color map for the landmask
    colors_mask = [(0.58, 0.747, 0.972),
            (1, 1, 1)]  # Grey (RGB for white)  # Light blue (RGB for light sky blue)
    # Create the colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreyToBlue", colors_mask)
    colors_mask = plt.get_cmap(custom_cmap, 2)

    # First subplot
    scatter1 = axes[0].scatter(x.ravel(), y.ravel(),marker='.',s=0.1,c=mask_interpol, cmap=colors_mask)
    axes[0].scatter(positions_ini[0,:], positions_ini[1,:], c=labels, cmap=colors, vmin=0, vmax=n_clusters-1)
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].set_title("Initial distribution of the clusters")
    axes[0].set_xlim(xmin-0.05*(xmax-xmin),xmax+0.05*(xmax-xmin)) 
    axes[0].set_ylim(ymin-0.05*(xmax-xmin),ymax+0.05*(ymax-ymin))  
    axes[0].set_aspect('equal', 'box')
    
    # Second subplot
    scatter2 = axes[1].scatter(x.ravel(), y.ravel(),marker='.',s=0.1,c=mask_interpol, cmap=colors_mask)
    axes[1].scatter(positions_end[0,:], positions_end[1,:], c=labels, cmap=colors, vmin=0, vmax=n_clusters-1)
    axes[1].set_xlabel("X Position")
    axes[1].set_title("Final distribution of the clusters")
    axes[1].set_aspect('equal', 'box')
    axes[1].set_xlim(xmin-0.08*(xmax-xmin),xmax+0.08*(xmax-xmin)) 
    axes[1].set_ylim(ymin-0.08*(xmax-xmin),ymax+0.08*(ymax-ymin))  
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cluster_names[j],
                            markerfacecolor=colors(j), markersize=8) for j in range(n_clusters)]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

    # Main title
    plt.suptitle(f"{n_clusters}clusters_{e}spars", fontsize=16)
    plt.subplots_adjust(right=0.8) 
    # Save the figure
    plt.savefig(f"{img_name}")
    
def gif_clusters_landmask_in(Fmap, n_clusters, labels, img_name, e, x, y, mask_interpol, freq):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define color map for clusters
    colors = plt.get_cmap("tab20", n_clusters)
    cluster_names = [f"Cluster {i + 1}" for i in range(n_clusters)]

    # Define color map for the landmask
    colors_mask = [(0.58, 0.747, 0.972), (1, 1, 1)]  # Light blue and white
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreyToBlue", colors_mask)
    colors_mask = plt.get_cmap(custom_cmap, 2)

    # Get axis limits from Fmap
    ymax = Fmap[:, 1, :].max()
    ymin = Fmap[:, 1, :].min()
    xmax = Fmap[:, 0, :].max()
    xmin = Fmap[:, 0, :].min()

    # Plot the static landmask once
    landmask_plot = ax.scatter(x.ravel(), y.ravel(), marker='.', s=0.1, c=mask_interpol, cmap=colors_mask)

    # Set axis limits and labels
    ax.set_xlim(xmin - 0.08 * (xmax - xmin), xmax + 0.08 * (xmax - xmin))
    ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + 0.08 * (ymax - ymin))
    ax.set_xlabel("Degrees W")
    ax.set_ylabel("Degrees E")
    ax.set_aspect('equal', 'box')

    # Create the legend once
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cluster_names[j],
                          markerfacecolor=colors(j), markersize=8) for j in range(n_clusters)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1))

    # Prepare dynamic scatter plot for clusters
    scatter = ax.scatter([], [], c=[], cmap=colors)

    def animate(i):
        # Update cluster positions and colors for frame `i`
        scatter.set_offsets(Fmap[i,:,:])  # Update (x, y) positions
        scatter.set_array(labels)      # Update cluster colors
        ax.set_title(f"Cluster at timestep {i + 1}")
        return scatter,

    # Filter frames to include only multiples of freq
    frames_to_plot = [i for i in range(Fmap.shape[0]) if i % freq == 0]

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=frames_to_plot, repeat=False, blit=True)

    # Adjust layout to fit legend
    plt.subplots_adjust(right=0.8)

    # Save the animation as a GIF
    ani.save(img_name, writer='pillow', fps=10)
    print(f"Animation saved to {gif_path}")

    plt.close(fig)

def gif_clusters_landmask(Fmap, n_clusters, labels, img_name, e, x, y, mask_interpol, freq):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define color map for clusters
    colors = plt.get_cmap("tab20", n_clusters)
    cluster_names = [f"Cluster {i + 1}" for i in range(n_clusters)]

    # Define color map for the landmask
    colors_mask = [(0.58, 0.747, 0.972), (1, 1, 1)]  # Light blue and white
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreyToBlue", colors_mask)
    colors_mask = plt.get_cmap(custom_cmap, 2)

    # Get axis limits from Fmap
    ymax = Fmap[:, 1, :].max()
    ymin = Fmap[:, 1, :].min()
    xmax = Fmap[:, 0, :].max()
    xmin = Fmap[:, 0, :].min()

    # Plot the static landmask once
    ax.scatter(x.ravel(), y.ravel(), marker='.', s=0.1, c=mask_interpol, cmap=colors_mask)

    # Set axis limits and labels
    ax.set_xlim(xmin - 0.08 * (xmax - xmin), xmax + 0.08 * (xmax - xmin))
    ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + 0.08 * (ymax - ymin))
    ax.set_xlabel("Degrees W")
    ax.set_ylabel("Degrees E")
    ax.set_aspect('equal', 'box')

    # Create the legend once
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cluster_names[j],
                          markerfacecolor=colors(j), markersize=8) for j in range(n_clusters)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1))

    def animate(i):
        # Remove previous trajectory scatter plots
        # Keep only the first collection (landmask)
        while len(ax.collections) > 1:
            ax.collections[-1].remove()

        # Plot the trajectories for frame `i`
        scatter = ax.scatter(Fmap[i, 0, :], Fmap[i, 1, :], c=labels, cmap=colors, vmin=0, vmax=n_clusters-1)
        ax.set_title(f"Cluster at timestep {i + 1}")
        return scatter,

    # Filter frames to include only multiples of freq
    frames_to_plot = [i for i in range(Fmap.shape[0]) if i % freq == 0]

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=frames_to_plot, repeat=False, blit=True)

    # Adjust layout to fit legend
    plt.subplots_adjust(right=0.8)

    # Save the animation as a GIF
    ani.save(img_name, writer='pillow', fps=10)
    print(f"Animation saved to {img_name}")

    plt.close(fig)

