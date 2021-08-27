import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
import json
import time

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
logo_path = dir_path + '/br.png'

# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#custom-describe
from functools import partial

q_25 = partial(pd.Series.quantile, q=0.25)
q_25.name = "25%"
q_75 = partial(pd.Series.quantile, q=0.75)
q_75.name = "75%"

LLOG_ERROR = '0'
# measurement data
LLOG_DATA = '1'
# application-specific configuration information
LLOG_CONFIG = '2'
# calibration data
LLOG_CALIBRATION = '4'
# read only memory + factory calibration and serialization type information
LLOG_ROM = '5'
# information/notes
LLOG_INFO = '6'
LLOG_NONE= '666'

# https://stackoverflow.com/questions/47466255/subclassing-a-pandas-dataframe-updates
class LLogSeries(pd.Series):
    _metadata = ['meta']

    @property
    def _constructor(self):
        return LLogSeries

    @property
    def _constructor_expanddim(self):
        return LLogDataFrame

    def pplot(self, d2=None, *args, **kwargs):
        columns = self.meta['columns']
        meta = {}
        for c in columns:
            if self.name == c.get('llabel'):
                meta = c
                break

        kwargs2 = kwargs | {'label': self.name}
        kwargs2 = kwargs2 | {'rasterized': True}

        for opt in ["color", "style", "label", "colormap"]:
            try:
                kwargs2 = kwargs2 | {opt:meta[opt]}
            except KeyError as e:
                # print(e)
                pass

        self.plot(*args, **kwargs2)
        plt.legend()
        plt.ylabel(meta.get('units'))

        if d2 is not None:
            plt.twinx()
            d2.pplot(*args, **kwargs)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(self.format_time_ticks)

        plt.grid(True)

        plt.tight_layout()

    def stats(self):
        return self.agg(["count", "mean", "std", "min", q_25, "median", q_75, "max"])

    @staticmethod
    @matplotlib.ticker.FuncFormatter
    def format_time_ticks(x, pos):
        """ Convert timedelta64[ns] duration into H:M:S format for plotting.

        Idea sourced from https://stackoverflow.com/a/42220184

        """
        seconds = x / 10**9 # convert nanoseconds to seconds
        return str(datetime.timedelta(seconds=seconds))


# https://stackoverflow.com/questions/48325859/subclass-pandas-dataframe-with-required-argument
class LLogDataFrame(pd.DataFrame):
    _metadata = ['meta']

    def __init__(self, *args, **kwargs):
        self.meta = kwargs.pop('meta', None)
        super().__init__(*args, **kwargs)

        # self.meta is not None:
        # sometimes dataframe meta is not provided in intermediate operations
        # so we need to check if self.meta exists
        # rename column index labels, and change column
        # series types according to metadata
        #
        # 2 in self.columns:
        # this is a hack, and should find a better way
        # we only want to convert the type during initial construction of the datafame
        # to do that, we are checking that we have not renamed the columns yet
        # this prevents converting the type in resulting copies or during intermediate operations
        # ex for results of astype() or logical operations
        # i'm not sure what i've created here, but to follow what's going on, look at the order of
        # LLogDataFrame.__init__, LLogDataFrame._constructor, and LLogSeries.expanddim
        if self.meta is not None and 2 in self.columns:
            # get column metadata
            metaColumns = self.meta.get('columns', [])

            # metadata or dataframe columns, which is shortest?
            l = min(len(metaColumns), len(self.columns))

            for n in range(l):
                llabel = metaColumns[n]['llabel']
                # we add 2 to the index because we drop the first two columns: time and llKey
                self.rename(columns={n+2:llabel}, inplace=True)

                try:
                    # our columns index name
                    name = self.columns[n]
                    # find column metadata by name
                    # there is probably a better way to format the metadata to make this easier
                    meta = [x for x in metaColumns if x['llabel'] == name][0]
                except IndexError as e:
                    # no metadata
                    continue

                try:
                    dtype = meta['dtype']
                    if dtype == "int":
                        self[name] = self[name].astype(int)
                    elif dtype == "float":
                        self[name] = self[name].astype(float)
                except KeyError as e:
                    try:
                        # assume float/numeric data
                        self[name] = self[name].astype(float)
                    except Exception as e:
                        # if it's a string it will fail, and the type will be object
                        pass


    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            df = LLogDataFrame(*args, meta=self.meta, **kwargs)
            return df
        return _c

    @property
    def _constructor_sliced(self):
        return LLogSeries

    def pplot(self, *args, **kwargs):
        d2 = kwargs.pop('d2', None)
        for c in self:
            self[c].pplot(*args, **kwargs)
        # plot d2 dataframe on secondary axis
        if d2 is not None:
            plt.twinx()
            d2.pplot(*args, **kwargs)

    def ttable(self, rl=False, *args, **kwargs):
        if rl is True:
            kwargs['rowLabels'] = self.index

        # rasterized = True...
        t = plt.table(cellText=self.to_numpy(dtype=str), colLabels=self.columns, loc='bottom', cellLoc='center', bbox=[0,0,1,1], *args, **kwargs)

        t.auto_set_font_size(False)
        t.set_fontsize(8)

        # row labels add an extra column
        min = rl*-1

        # for each row find the max number of lines in that row...
        # iterate rows..
        for i in range(len(self.index) + 1):
            text = t[i,0].get_text().get_text()
            lines = text.count('\n') + 1

            # iterate columns and find the max height...
            maxLines = 0
            for j in range(min, len(self.columns)):
                # there is no row label for the row of cells holding the column labels
                if i == 0 and j == -1:
                    continue
                text = t[i,j].get_text().get_text()
                lines = text.count('\n')
                if lines > maxLines:
                    maxLines = lines

            # iterate columns and set the height...
            for j in range(min, len(self.columns)):
                # there is no row label for the row of cells holding the column labels
                if i == 0 and j == -1:
                    continue
                t[i,j].set_height(maxLines + 1)

        plt.axis('off')
        plt.title(self.meta['llType'])

        plt.tight_layout()

    def stats(self):
        stats = self.agg(["count", "mean", "std", "min", q_25, "median", q_75, "max"])
        return LLogDataFrame(stats, meta=self.meta)


class LLogReader:
    def __init__(self, logfile, metafile):
        self.df = pd.read_csv(logfile, sep=' ', header=None).dropna(axis='columns', how='all').set_index(0, drop=True).sort_index()
        with open(metafile, 'r') as f:
            self.meta = json.load(f)

        # todo move this to LLogDataFrame constructor
        # or remove these columns from the logdataframe completely
        self.df.rename(columns={1:'llKey'}, inplace=True)
        self.df['llKey'] = self.df['llKey'].astype(int)
        # convert times to timestamps
        self.df.index = pd.to_datetime(self.df.index, unit='s')
        # convert timestamps to timedeltas (duration from start of test)
        self.df.index -= self.df.index[0]

        for llKey, llDesc in self.meta.items():
            DF = self.df
            value = DF[DF['llKey'] == int(llKey)]
            if value.size == 0:
                # no log entries for this llType
                continue

            # drop any entirely N/A columns
            value = value.dropna(axis='columns', how='all')
            # drop the llKey column
            value = value.drop('llKey', axis=1)

            # eg for each llType name in log, set self.type to
            # the dataframe representing only that type
            value = LLogDataFrame(value, meta=llDesc)
            llType = llDesc['llType']
            setattr(self, llType, value)

    def figure(self, height_ratios=[1,4,4], columns=2, suptitle='', header='', footer=''):
        f = plt.figure(figsize=(8.5, 11.0))
        plt.suptitle(suptitle)
        footer_buffer_ratio = sum(height_ratios) * 0.02
        height_ratios.append(footer_buffer_ratio)
        rows = len(height_ratios)
        spec = f.add_gridspec(rows, columns, height_ratios=height_ratios)
        f.text(0.98, 0.98, header, size=8, horizontalalignment='right', verticalalignment='bottom')
        f.text(0.98, 0.02, footer, size=8, horizontalalignment='right')

        # add a line, using our footer buffer subplot (the line is plotted in figure coords)
        f.add_subplot(spec[-1,:])
        plt.plot([0.35, 0.65], [0.02, 0.02], color='#2c99ce', lw=3, clip_on=False, transform=f.transFigure)
        plt.axis('off')

        # add br logo image
        im = plt.imread(logo_path)
        f.figimage(im, 2, 2)
        return f, spec


class LLogWriter:
    def __init__(self, metafile, logfile=None, console=True):
        with open(metafile, 'r') as f:
            self.meta = json.load(f)
        self.logfile = logfile
        self.console = console

        if self.logfile:
            # todo error if it exists
            self.logfile = open(self.logfile, 'w')
            # this is a hack
            # pandas will have an error if any row has
            # fewer columns than the first row
            self.log(LLOG_NONE, '                                                 ')

    def log(self, type, data, t=None):
        if t is None:
            t = time.time()
        logstring = f'{t:.6f} {type} {data}\n'
        if self.console:
            print(logstring, end='')
        if self.logfile:
            self.logfile.write(logstring)

    def close(self):
        if self.logfile:
            self.logfile.close()
