import rasterio as rio
from localkriging import mpiops


class RasterWriter:

    def __init__(self, output_tif, profile):
        """
        :param output_tif: str
            output file name
        :param profile: rio.profile.Profile instance
        """
        self.output = output_tif

    def write(self, data, window):
        if mpiops.rank == 0:
            with open(self.output, 'w') as dst:
                pass
        else:
            mpiops.comm.send(data, dest=0)



