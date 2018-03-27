import logging
import rasterio as rio
from rasterio.windows import Window
from localkriging import mpiops
log = logging.getLogger(__name__)


class RasterWriter:

    def __init__(self, output_tif, kriged_residuals, profile):
        """
        :param output_tif: str
            output file name
        :param kriged_residuals: str
            kriged residual output file name
        :param profile: rio.profile.Profile instance
        """

        if mpiops.rank == 0:
            self.dst = rio.open(output_tif, 'w', **profile)
            self.dst_residuals = rio.open(kriged_residuals, 'w', **profile)

    def write(self, data_win_dict):
        """
        :param data_win_dict: dict
            dictionary of data, residual and window
        """
        if mpiops.rank == 0:
            for r in range(mpiops.size):
                data_win_dict = mpiops.comm.recv(source=r) \
                    if r != 0 else data_win_dict
                data = data_win_dict['data']
                res = data_win_dict['residuals']
                window = Window(*data_win_dict['window'])
                self.dst.write(data.astype(rio.float32),
                               window=window, indexes=1)
                self.dst_residuals.write(res.astype(rio.float32),
                                         window=window, indexes=1)
                log.info('Finished writing partition in process {}'.format(r))

        else:
            mpiops.comm.send(data_win_dict, dest=0)
