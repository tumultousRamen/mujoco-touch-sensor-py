import ctypes
import numpy as np
import mujoco as mj
from mujoco import MjModel, MjData, MjvScene, MjvOption

# Checks if a plugin attribute attribute exists
# Input: input_str (string)
# Output: True if the attribute exists, False otherwise
def check_attr(input_str):
    value = ''.join(char for char in input_str if not char.isspace())
    try:
        float(value)
        return True
    except ValueError:
        return False

class TouchGridPy:
    @staticmethod
    def create(model, data, instance):
        # Check if required attributes are present
        required_attrs = ["gamma", "nchannel", "size", "fov"]
        for attr in required_attrs:
            if not check_attr(mj.mj_getPluginConfig(model, instance, attr)):
                raise ValueError(f"Missing {attr} in touch_grid sensor plugin configuration")

        # nchannel
        nchannel = int(mj.mj_getPluginConfig(model, instance, "nchannel"))
        if nchannel < 1 or nchannel > 6:
            raise ValueError("nchannel must be between 1 and 6")

        # size
        size = [int(x) for x in mj.mj_getPluginConfig(model, instance, "size").split()]
        if len(size) != 2:
            raise ValueError("Both horizontal and vertical resolutions must be specified")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("Horizontal and vertical resolutions must be positive")

        # field of view
        fov = [float(x) for x in mj.mj_getPluginConfig(model, instance, "fov").split()]
        if len(fov) != 2:
            raise ValueError("Both horizontal and vertical fields of view must be specified")
        if fov[0] <= 0 or fov[0] > 180:
            raise ValueError("fov[0] must be a float between (0, 180] degrees")
        if fov[1] <= 0 or fov[1] > 90:
            raise ValueError("fov[1] must be a float between (0, 90] degrees")

        # gamma
        gamma = float(model.plugin_config[instance]["gamma"])
        if gamma < 0 or gamma > 1:
            raise ValueError("gamma must be a nonnegative float between [0, 1]")

        # Create and return the TouchGridPy instance
        return TouchGridPy(model, data, instance, nchannel, size, fov, gamma)

    def __init__(self, model, data, instance, nchannel, size, fov, gamma):
        self.nchannel = nchannel
        self.size = np.array(size)
        self.fov = np.array(fov)
        self.gamma = gamma

        # Make sure sensor is attached to a site
        for i in range(model.nsensor):
            if model.sensor_type[i] == 'plugin' and model.sensor_plugin[i] == instance:
                if model.sensor_objtype[i] != 'site':
                    raise ValueError("Touch Grid sensor must be attached to a site")

        # Allocate distance array
        self.distance = np.zeros(size[0] * size[1])

        # Initialize other necessary attributes
        self._calculate_bin_edges()
        self.taxel_positions = self._calculate_taxel_positions()

    def _calculate_bin_edges(self):
        # Bin edge calculation (this implementation replicates BinEdges in touch_grid.cc)
        def fovea(x, gamma):
            return gamma * x**5 + (1 - gamma) * x

        self.x_edges = np.linspace(-1, 1, self.size[0] + 1)
        self.y_edges = np.linspace(-1, 1, self.size[1] + 1)
        
        self.x_edges = fovea(self.x_edges, self.gamma)
        self.y_edges = fovea(self.y_edges, self.gamma)
        
        self.x_edges *= self.fov[0] * mj.mjPI / 180
        self.y_edges *= self.fov[1] * mj.mjPI / 180

    def _calculate_taxel_positions(self):
        # Calculate center of each taxel in spherical coordinates
        # Refer to Visualize in touch_grid.cc for the comprehesive implementation:
        x_centers = (self.x_edges[1:] + self.x_edges[:-1]) / 2
        y_centers = (self.y_edges[1:] + self.y_edges[:-1]) / 2
        xx, yy = np.meshgrid(x_centers, y_centers)
        
        # Convert to Cartesian coordinates (assuming unit sphere)
        x = np.cos(yy) * np.sin(xx)
        y = np.sin(yy)
        z = -np.cos(yy) * np.cos(xx)
        
        return np.stack([x, y, z], axis=-1)

    def reset(self, model, instance):
        # Reset distance array
        self.distance.fill(0)

    def compute(self, model, data, instance):
        # Placeholder for compute logic
        # This will need to be implemented based on the C++ version
        pass

    def visualize(self, model, data, opt, scn, instance):
        # Placeholder for visualization logic
        # This will need to be implemented based on the C++ version
        pass

    def get_taxel_data(self):
        return {
            'positions': self.taxel_positions,
            'distances': self.distance.reshape(self.size),
            'x_edges': self.x_edges,
            'y_edges': self.y_edges
        }

    def get_taxel_forces(self):
        # Placeholder for force data
        # This would be populated in the compute method
        forces = np.zeros((self.size[0], self.size[1], self.nchannel))
        return forces

def touch_grid_compute(model, data, instance):
    # This function will be called by MuJoCo
    touch_grid = TouchGridPy(model, data, instance)
    touch_grid.compute(model, data, instance)
    # Here we would need to copy the computed data back to MuJoCo's data structure

def register_plugin():
    # This is a placeholder for the actual registration process
    pass

if __name__ == "__main__":
    register_plugin()