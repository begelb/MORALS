from geomstats.geometry.hypersphere import Hypersphere
import numpy as np
import os
import plotly.graph_objs as go
import math

n_size = 500
theta = np.random.uniform(low= 0 * -3.14159, high = 2 * 3.14159, size = n_size)
phi = np.random.uniform(low= 0 * 3.14159, high = 1 * 3.14159, size = n_size)
w = np.random.uniform(low = -1, high = 1, size = n_size)

# this would just be used for plotting
input = np.concatenate((theta.reshape(-1,1), phi.reshape(-1,1)), axis = 1)

def sphere_map(input):
    theta, phi = input
    phi1 =  np.pi/2 * ( np.cbrt(2*phi/np.pi -1) + 1)
    theta1 = theta + np.pi * np.sin(phi)
    return (theta1, phi1)

def Euclidean_dynamics(theta, phi, w):
    x = np.cos(theta)*np.sin(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(phi)
    w/=2
    return (x, y, z, w)

def plot_data(x, y, z, i):
    trace = go.Scatter3d(
    x = x, y = y, z = z, mode = 'markers', marker = dict(
        size = 3
        )
    )
    layout = go.Layout(title = '3D Scatter plot')
    fig = go.Figure(data = [trace], layout = layout)
    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-1,1],),
                        yaxis = dict(nticks=4, range=[-1,1],),
                        zaxis = dict(nticks=4, range=[-1,1],),),
        margin=dict(r=20, l=10, b=10, t=10))
  #  fig.show()
    file_name = f'sphere_data_plots/{i}.png'
    fig.write_image(file_name) 

def interate_sphere_map(k, input, w_in):
    for i in range(k):
        theta, phi = sphere_map(input)
        x, y, z, w_out = Euclidean_dynamics(theta, phi, w_in)
        input = (theta, phi)
        plot_data(x, y, z, i)

        for j in range(len(x)):
            file_name = f'examples/data/sphere/{j}.txt'

            with open(file_name, "a") as f:
                print(x[j], y[j], z[j], w_out[j], sep=", ", file=f)

        if i+1 == k:
            file_name = 'examples/data/sphere_labels.txt'
            with open(file_name, "w") as f:
                for j in range(len(x)):

                    # Here I am assuming the fact that all of the other coordinates are close to zero and therefore not checking
                    if math.isclose(z[j], 1, rel_tol=0.05):
                        label = 1
                    elif math.isclose(z[j], -1, rel_tol=0.05):
                        label = 0
                    else:
                        raise Exception('point does not have label')
                    print(f'{j}.txt', label, sep=",", file=f)


    return input

output_dir = "examples/data/sphere/"
os.makedirs(output_dir, exist_ok=True)

# clear previously made data files
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path): 
        open(file_path, "w").close()


interate_sphere_map(5, (theta, phi), w)
