import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_3d(c,cmap=plt.cm.jet,bar_label='wave speed (km/s)', scale=10,vmin=None,vmax=None):
    grid_size=c.shape[0]
    nx,ny,nz=grid_size,grid_size,grid_size
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), -np.arange(nz))
    data = np.transpose(c,[1,0,2])*scale
    #data=np.array(c)
    if vmin==None:
        vmin=data.min()
    if vmax==None:
        vmax=data.max()
    kw = {
        'vmin': vmin,
        'vmax': vmax,
    }
    norm = mpl.colors.Normalize(**kw)
    fig = plt.figure(figsize=(8, 6))
    ax0 = fig.add_subplot(121, projection='3d',computed_zorder=False)
    # Plot contour surfaces
    ax0.plot_surface(
        X[:, -1, :], Y[:, -1, :], Z[:,-1,:],rstride=1, cstride=1, facecolors=cmap(norm(data[:, -1, :])),shade=False)
    ax0.plot_surface(
        X[:, :, 0], Y[:, :, 0], Z[:,:,0],rstride=1, cstride=1, facecolors=cmap(norm(data[:, :, 0])),shade=False)

    ax0.plot_surface(
        X[0, :, :], Y[0, :, :], Z[0,:,:],rstride=1, cstride=1, facecolors=cmap(norm(data[0, :, :])),shade=False)
    # # Plot edges
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax0.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    edges_kw = dict(color='0.2', linewidth=2, zorder=1e2)
    ax0.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax0.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax0.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax0.set(
        xlabel='X (km)',
        ylabel='Y (km)',
        zlabel='Z (km)',
        zticks=[0,-int(nz/4),-int(nz/4*2),-int(nz/4*3),-nz],
        zticklabels=[0,-25, -50,-75,-100],
        xticks=[0,int(nx/4),int(nx/4*2),int(nx/4*3),nx],
        xticklabels=[0, 25,50,75,100],
        yticks=[0,int(nx/4),int(ny/4*2),int(ny/4*3),ny],
        yticklabels=[0, 25,50,75,100],
    )

    ax0.view_init(10, -30, 0)
    ax0.set_box_aspect(None, zoom=1)
    #ax.set_title('Front')


    ax1 = fig.add_subplot(122, projection='3d',computed_zorder=False)

    # Plot contour surfaces
    ax1.plot_surface(
        X[:, :, -1], Y[:, :, -1], Z[:,:,-1],rstride=1, cstride=1, facecolors=cmap(norm(data[:, :, -1])),shade=False)
    ax1.plot_surface(
        X[-1, :, :], Y[-1, :, :], Z[-1,:,:],rstride=1, cstride=1, facecolors=cmap(norm(data[-1, :, :])),shade=False)
    ax1.plot_surface(
        X[:, 0, :], Y[:, 0, :], Z[:,0,:],rstride=1, cstride=1, facecolors=cmap(norm(data[:, 0, :])),shade=False)


    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax1.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color='0.2', linewidth=2, zorder=1e2)
    ax1.plot([xmin, xmin], [ymin, ymax], [zmin,zmin], **edges_kw)
    ax1.plot([xmin, xmax], [ymax, ymax], [zmin,zmin], **edges_kw)
    ax1.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], **edges_kw)



    # Set labels and zticks
    ax1.set(
        xlabel='X (km)',
        ylabel='Y (km)',
        zlabel='Z (km)',
        zticks=[0,-int(nz/4),-int(nz/4*2),-int(nz/4*3),-nz],
        zticklabels=[0,-25, -50,-75,-100],
        xticks=[0,int(nx/4),int(nx/4*2),int(nx/4*3),nx],
        xticklabels=[0, 25,50,75,100],
        yticks=[0,int(nx/4),int(ny/4*2),int(ny/4*3),ny],
        yticklabels=[0, 25,50,75,100],
    )
    cax = ax1.inset_axes([1.25, 0.2, 0.05, 0.6])
    ax1.view_init(10, -30, 0)
    ax1.set_box_aspect(None, zoom=1)
    #ax1.set_title('Back',y=-0.1,fontsize=18)
    ax0.grid(False)
    ax1.grid(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1,cax=cax,label=bar_label, fraction=0.035)
    return [fig,ax0,ax1]






