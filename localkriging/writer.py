import rasterio as rio
from rasterio.windows import Window
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

        if mpiops.rank == 0:
            self.dst = rio.open(self.output, 'w', **self.profile)

    def write(self, data_win_dict):
        if mpiops.rank == 0:
            for r in range(mpiops.size):
                data_win_dict = mpiops.comm.recv(source=r) \
                    if r != 0 else data_win_dict
                data = data_win_dict['data']
                if data is not None:
                    window = Window(*data_win_dict['window'])
                    self.dst.write(data.astype(rio.float32),
                                   window=window, indexes=1)
        else:
            mpiops.comm.send(data_win_dict, dest=0)
