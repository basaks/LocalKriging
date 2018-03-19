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
        self.profile = profile
        self.dst = rio.open(self.output, 'w', **self.profile)

    def write(self, data, window, indexes=1):
        if mpiops.rank == 0:
            print('writing in ', window)
            self.dst.write(data, window=window, indexes=indexes)
        else:
            mpiops.comm.send(data, dest=0)
