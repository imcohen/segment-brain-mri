import numpy as np
import plotly.graph_objects as go
from skimage import io


def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }


def plot_stack(vol):
    volume = vol
    r, c = volume[0].shape[:2]

    # Define frames
    nb_frames = volume.shape[0]

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((nb_frames-1)/10 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-1 - k]),
        cmin=0, cmax=200
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(nb_frames-1)/10 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[-1]),
        colorscale='Gray',
        cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
        ))

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             width=600,
             height=600,
             scene=dict(
                        zaxis=dict(range=[-0.1, (nb_frames)/10], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    fig.show()


def read_mri(df, patient_id):
    sub_df = df[df.patient_id == patient_id].reset_index()
    dim = len(sub_df)
    image_stack = np.zeros((dim, 256, 256, 3), dtype='uint8')
    mask_stack = np.zeros((dim, 256, 256), dtype='uint8')
    for i in range(dim):
        image_stack[i] = io.imread(sub_df.image_path[i])
        mask_stack[i] = io.imread(sub_df.mask_path[i])
    return image_stack, mask_stack, list(sub_df[sub_df.has_tumor == 1].index)
