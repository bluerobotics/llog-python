#!/usr/bin/python3

import argparse
import llog as ll
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

meta = dir_path + '/test-llog.meta'
output = dir_path + '/test-llog.csv'
log = ll.LLogWriter(meta, output, console=False)

for i in range(20):
    log.log(ll.LLOG_ERROR, f'error{i}')

for i in range(100):
    log.log(ll.LLOG_DATA, f'{math.sin(i/5.0):0.2f} {i*i} {2*math.sin(i/4.0):0.2f} {math.sin(i/4.0 + 2):0.2f} {math.sin(i/4.0 + 4):0.2f}')

for i in range(5):
    log.log(ll.LLOG_ROM, f'{1.0/(i+1):0.6f} {i*100}')

log.close()

# open/read the log file
log = ll.LLogReader(output, meta)

dp = log.data['p']
dt = log.data['t']
dg = log.data[['gx', 'gy', 'gz']]

header = 'test-llog.py report header'
footer = 'test-llog.py report footer'
f, spec = log.figure(height_ratios=[2,2,4,2], columns=3, suptitle='some data plots', header=header, footer=footer)

plt.subplot(spec[0,:])
log.rom.ttable()

plt.subplot(spec[1,0])
dp.pplot(d2=dt, title='pressure + temperature')

plt.subplot(spec[1,1:])
dt.pplot(title='temperature data')

plt.subplot(spec[2,:])
dg.pplot(title='gyro data')

plt.subplot(spec[3,0])
dg.gx.pplot(title='gyro x')

plt.subplot(spec[3,1])
dg.gy.pplot(title='gyro y')

plt.subplot(spec[3,2])
dg.gz.pplot(title='gyro z')

with PdfPages('test.pdf') as pdf:
    pdf.savefig()
    f, spec = log.figure(height_ratios=[2,6], columns=1, suptitle='gyro data head', header=header, footer=footer)
    plt.subplot(spec[0])
    dgs = dg.stats()
    dg.stats().ttable(rl=True)

    plt.subplot(spec[1])
    dg.head(20).ttable(rl=True)
    pdf.savefig()

    f, spec = log.figure(height_ratios=[1], columns=1, suptitle='log errors', header=header, footer=footer)
    plt.subplot(spec[0])
    log.error.ttable(rl=True)
    pdf.savefig()

plt.show()
